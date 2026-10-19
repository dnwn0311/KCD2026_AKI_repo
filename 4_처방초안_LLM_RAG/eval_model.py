# -*- coding: utf-8 -*-
"""
5단계: test 데이터로 모델 평가 (기준선과 같은 지표)

  python eval_model.py --adapter runs/qlora_aki/adapter         (파인튜닝 모델)
  python eval_model.py --no-adapter --tag zeroshot --limit 200  (학습 전 원본 모델)

출력: results/eval_<tag>.json (집계만), _work/preds_<tag>.jsonl (환자 단위 예측, PC 밖으로 내보내지 않음)
"""
import argparse
import json
import os
import time

import torch

import common as C
import llm_common as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default=os.path.join(C.HERE, "runs", "qlora_aki", "adapter"))
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()
    tag = a.tag or ("zeroshot" if a.no_adapter else os.path.basename(os.path.dirname(a.adapter.rstrip("/\\"))))

    t0 = time.time()
    dtype = L.pick_dtype()
    use_4bit = not a.no_4bit
    tok = L.load_tokenizer(a.model if a.no_adapter else a.adapter)
    tok.padding_side = "left"
    model = L.load_model(a.model, use_4bit, dtype)
    if not a.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    if not use_4bit and torch.cuda.is_available():
        model.to("cuda")
    device = next(model.parameters()).device

    rows = L.read_jsonl(os.path.join(C.DATA, "test.jsonl"), a.limit or None)
    stops = L.end_token_ids(tok)
    preds, results = [], []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        prompts = [L.render_prompt(tok, r["messages"][:-1]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 eos_token_id=stops or None, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(chunk, gen):
            text = tok.decode(g, skip_special_tokens=True)
            p = C.parse_answer(text)
            results.append((r["scenario"], r["label"], p))
            preds.append({"scenario": r["scenario"], "label": r["label"], "pred": p, "raw": text})
        print("  %d/%d" % (min(i + a.batch, len(rows)), len(rows)), flush=True)

    os.makedirs(C.WORK, exist_ok=True)
    with open(os.path.join(C.WORK, "preds_%s.jsonl" % tag), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    res = C.set_metrics(results)
    res["tag"], res["model"] = tag, a.model
    res["adapter"] = None if a.no_adapter else a.adapter
    res["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(res, os.path.join(C.RESULTS, "eval_%s.json" % tag))
    o = res["overall"]
    print("[%s] JSON 해석 %.1f%%, 전체 micro F1 %.3f (신규 %.3f, 감량 %.3f, 증량 %.3f, 중단 %.3f), 대표약 재현율 %s"
          % (tag, 100 * (o["parse_rate"] or 0), o["micro_f1"], *[o["by_action"][k]["f1"] or 0 for k in C.ACTIONS],
             o["target_recall"]))


if __name__ == "__main__":
    main()
