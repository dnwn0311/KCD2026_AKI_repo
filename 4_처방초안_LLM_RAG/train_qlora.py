# -*- coding: utf-8 -*-
"""
4단계: QLoRA 파인튜닝 (기본: Qwen/Qwen3.5-9B, 4비트, 16GB GPU 기준)

- 입력(system+user) 부분은 손실 계산에서 제외하고, 답변(JSON) 부분만 학습
- 생각 모드(<think>)는 끈 형식으로 학습
- 중간 체크포인트를 저장하므로, 끊기면 --resume 으로 이어서 학습

사용:
  python train_qlora.py                         (기본 설정)
  python train_qlora.py --limit-train 4000      (빨리 시험)
  python train_qlora.py --resume                (이어서 학습)
  python train_qlora.py --model Qwen/Qwen3-8B   (다른 모델)
"""
import os as _os
if _os.name == "nt":
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.8"
import argparse
import glob
import inspect
import json
import os
import time

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback

import common as C
import llm_common as L


class SFTData(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items, self.dropped = [], 0
        end = L.end_token(tok)
        for r in rows:
            prompt = L.render_prompt(tok, r["messages"][:-1])
            answer = r["messages"][-1]["content"] + end
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer, add_special_tokens=False)["input_ids"]
            ids = p_ids + a_ids
            if len(ids) > max_len:
                self.dropped += 1
                continue
            self.items.append((ids, [-100] * len(p_ids) + a_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


class AnswerOnlyTrainer(Trainer):
    """답변 구간의 로짓만 계산해 손실을 구한다.

    Qwen3.5 는 어휘가 매우 커서(약 25만) 전체 위치의 로짓을 만들면 16GB GPU 에서 메모리가 부족해진다.
    입력(프롬프트) 위치는 어차피 손실에서 제외되므로, 답변이 시작되는 위치부터 끝까지만 로짓을 만든다.
    """
    _keep_ok = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kw):
        labels = inputs.pop("labels")
        n = labels.shape[1]
        valid = labels != -100
        first = torch.where(valid.any(1), valid.float().argmax(1), torch.full_like(valid[:, 0], n - 1, dtype=torch.long))
        k = int(n - first.min().item() + 1)
        k = max(2, min(k, n))
        out = None
        if AnswerOnlyTrainer._keep_ok:
            try:
                out = model(**inputs, logits_to_keep=k)
            except TypeError:
                AnswerOnlyTrainer._keep_ok = False
                print("[알림] 이 모델은 logits_to_keep 를 지원하지 않아 전체 로짓으로 계산합니다.", flush=True)
        if out is None:
            out = model(**inputs)
        logits = out.logits[:, -k:, :]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, n - k + 1:]
        tgt = shift_labels.reshape(-1).to(shift_logits.device)
        loss_sum = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), tgt,
                                                     ignore_index=-100, reduction="sum")
        # 기울기 누적 시 Trainer 가 넘겨주는 전체 답변 토큰 수로 나눠야 손실·기울기 크기가 맞는다
        if num_items_in_batch is not None and not return_outputs:
            denom = num_items_in_batch.to(loss_sum.device) if torch.is_tensor(num_items_in_batch) \
                else torch.tensor(float(num_items_in_batch), device=loss_sum.device)
        else:
            denom = (tgt != -100).sum()
        loss = loss_sum / denom.clamp(min=1)
        return (loss, {"loss": loss}) if return_outputs else loss


class Status(TrainerCallback):
    """진행 상황을 results/train_status_<run>.json 에 기록 (원격 모니터링용, 집계만)"""

    def __init__(self, path):
        self.path, self.t0, self.s0 = path, time.time(), None
        self.last = {}

    def write(self, **kw):
        self.last.update(kw)
        self.last["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        C.save_json(self.last, self.path)

    def on_train_begin(self, args, state, control, **kw):
        self.s0 = state.global_step
        self.write(state="training", step=state.global_step, max_steps=state.max_steps)

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        done = state.global_step - (self.s0 or 0)
        el = time.time() - self.t0
        eta = (state.max_steps - state.global_step) * el / done / 3600 if done > 0 else None
        upd = dict(state="training", step=state.global_step, max_steps=state.max_steps,
                   epoch=round(state.epoch or 0, 3), session_hours=round(el / 3600, 2),
                   eta_hours=round(eta, 2) if eta is not None else None,
                   sec_per_step=round(el / done, 1) if done > 0 else None)
        if "loss" in logs:
            upd["last_loss"] = round(float(logs["loss"]), 4)
        if "eval_loss" in logs:
            upd["last_eval_loss"] = round(float(logs["eval_loss"]), 4)
        if torch.cuda.is_available():
            upd["gpu_mem_gb_peak"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        self.write(**upd)

    def on_save(self, args, state, control, **kw):
        self.write(last_checkpoint_step=state.global_step)


class LossLog(TrainerCallback):
    def __init__(self, path):
        self.path, self.hist = path, []

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if k in logs:
                    rec[k] = round(float(logs[k]), 6)
            self.hist.append(rec)
            C.save_json(self.hist, self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "qlora_aki"))
    ap.add_argument("--limit-train", type=int, default=8000, help="상황별로 고르게 뽑을 학습 표본 수 (0=전체)")
    ap.add_argument("--limit-val", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--no-4bit", action="store_true", help="4비트 양자화 끄기 (시험용)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--upcast", action="store_true", help="비양자화 층을 float32 로 (메모리 많이 씀)")
    ap.add_argument("--quiet", action="store_true", help="진행 막대 대신 10단계마다 한 줄 기록 (로그 파일용)")
    a = ap.parse_args()
    run = os.path.basename(os.path.normpath(a.out))
    status = Status(os.path.join(C.RESULTS, "train_status_%s.json" % run))
    status.write(state="loading", pid=os.getpid(), model=a.model)
    try:
        train(a, run, status)
    except BaseException as e:  # 오류·중단도 상태 파일에 남긴다 (자동 재시작 스크립트가 다시 실행)
        status.write(state="error", error="%s: %s" % (type(e).__name__, str(e)[:500]))
        raise


def train(a, run, status):

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    t0 = time.time()
    dtype = L.pick_dtype()
    use_4bit = not a.no_4bit
    if use_4bit and not torch.cuda.is_available():
        raise SystemExit("[오류] GPU(CUDA)를 찾지 못했습니다. check_env.py 로 환경을 확인하세요.")

    tok = L.load_tokenizer(a.model)
    train_rows = L.read_jsonl(os.path.join(C.DATA, "train.jsonl"), a.limit_train or None)
    val_rows = L.read_jsonl(os.path.join(C.DATA, "val.jsonl"), a.limit_val or None)
    train_ds, val_ds = SFTData(train_rows, tok, a.max_len), SFTData(val_rows, tok, a.max_len)
    lens = [len(x[0]) for x in train_ds.items]
    print("[데이터] 학습 %d건 (길이 초과 제외 %d), 검증 %d건, 평균 토큰 %.0f, 최대 %d"
          % (len(train_ds), train_ds.dropped, len(val_ds), sum(lens) / max(1, len(lens)), max(lens or [0])),
          flush=True)

    model = L.load_model(a.model, use_4bit, dtype)
    if use_4bit and a.upcast:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # prepare_model_for_kbit_training 은 임베딩·출력층(어휘 약 25만 x 4096)을 float32 로 올려
        # 그것만으로 약 8GB 를 쓴다. 16GB GPU 에서는 bf16 그대로 두고 체크포인팅만 켠다.
        for p_ in model.parameters():
            p_.requires_grad_(False)
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    regex, leafs = L.lora_target_regex(model)
    print("[LoRA] 대상 층: %s" % ", ".join(leafs), flush=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=regex, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(output_dir=a.out, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
              gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, max_steps=a.max_steps,
              learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
              save_steps=a.save_steps, save_total_limit=2, eval_steps=a.eval_steps,
              bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
              gradient_checkpointing=True, report_to=[], remove_unused_columns=False,
              dataloader_num_workers=0, prediction_loss_only=True, optim="paged_adamw_8bit" if use_4bit else "adamw_torch")
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if "gradient_checkpointing_kwargs" in params:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "save_strategy" in params:
        kw["save_strategy"] = "steps"
    if a.quiet:
        kw["disable_tqdm"] = True
    kw = {k: v for k, v in kw.items() if k in params}
    args = TrainingArguments(**kw)

    logcb = LossLog(os.path.join(C.RESULTS, "train_log_%s.json" % run))
    trainer = AnswerOnlyTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id), callbacks=[logcb, status])
    resume = bool(a.resume and glob.glob(os.path.join(a.out, "checkpoint-*")))
    status.write(state="training", resumed=resume)
    print("[학습] %s" % ("체크포인트에서 이어서 학습" if resume else "처음부터 학습"), flush=True)
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter = os.path.join(a.out, "adapter")
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    info = {"model": a.model, "train_examples": len(train_ds), "val_examples": len(val_ds),
            "dropped_too_long": train_ds.dropped, "max_len": a.max_len, "epochs": a.epochs,
            "lr": a.lr, "batch": a.batch, "grad_accum": a.grad_accum, "lora_r": a.lora_r,
            "lora_targets": leafs, "4bit": use_4bit, "hours": round((time.time() - t0) / 3600, 2)}
    C.save_json(info, os.path.join(C.RESULTS, "train_info_%s.json" % run))
    status.write(state="finished", adapter=adapter)
    print("[완료] 어댑터 저장: %s (%.2f시간)" % (adapter, info["hours"]), flush=True)


if __name__ == "__main__":
    main()
