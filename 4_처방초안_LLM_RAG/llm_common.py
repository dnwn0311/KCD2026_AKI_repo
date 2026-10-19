# -*- coding: utf-8 -*-
"""학습·평가 공용: 모델/토크나이저 불러오기, 채팅 형식 만들기"""
import json
import os
import random
import re

import torch
import transformers

import common as C


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(name):
    tok = transformers.AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def end_token(tok):
    tid = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(tid, int) and tid != tok.unk_token_id and tid is not None and tid >= 0:
        return "<|im_end|>"
    return tok.eos_token


def end_token_ids(tok):
    ids = []
    for t in ("<|im_end|>", tok.eos_token):
        if t:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id and i not in ids:
                ids.append(i)
    return ids


def render_prompt(tok, messages):
    """system+user 까지를 채팅 형식으로 만들고 답변 시작 위치에서 끝낸다 (생각 모드 끔)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _dtype_kw(dtype):
    """transformers 5.x 는 dtype, 4.x 는 torch_dtype"""
    import inspect
    params = inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters
    return {"dtype": dtype} if "dtype" in params or int(transformers.__version__.split(".")[0]) >= 5 \
        else {"torch_dtype": dtype}


def load_model(name, use_4bit, dtype):
    kw = dict(_dtype_kw(dtype))
    kw["low_cpu_mem_usage"] = True
    if use_4bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM",
                     "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(name, **kw)
            print("[모델] %s 로 불러옴 (%s)" % (cls_name, name), flush=True)
            return model
        except Exception as e:  # 클래스가 이 모델을 지원하지 않으면 다음 후보
            errors.append("%s: %s: %s" % (cls_name, type(e).__name__, str(e)[:300]))
    raise SystemExit("[오류] 모델을 불러오지 못했습니다.\n" + "\n".join(errors) +
                     "\n- transformers 가 이 모델을 지원하는지 check_env.py 로 확인하세요."
                     "\n- 안 되면 --model Qwen/Qwen3-8B 처럼 다른 모델로 시험할 수 있습니다.")


SKIP_PARTS = ("visual", "vision", "mtp", "merger", "lm_head", "embed", "patch")
# in_proj_a / in_proj_b: Qwen3.5 선형 어텐션의 아주 작은 투영층(헤드 수 크기)이라 제외
SKIP_LEAFS = {"gate", "router", "shared_expert_gate", "in_proj_a", "in_proj_b"}


def lora_target_regex(model):
    """언어 모델 부분의 Linear 층만 LoRA 대상으로 (비전 인코더·출력층 제외)."""
    leafs = set()
    for n, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            low = n.lower()
            if any(p in low for p in SKIP_PARTS):
                continue
            leaf = n.split(".")[-1]
            if leaf in SKIP_LEAFS:
                continue
            leafs.add(leaf)
    if not leafs:
        raise SystemExit("[오류] LoRA 를 붙일 Linear 층을 찾지 못했습니다.")
    leafs = sorted(leafs)
    regex = r"^(?!.*(visual|vision|mtp|merger)).*\.(%s)$" % "|".join(re.escape(x) for x in leafs)
    return regex, leafs


def read_jsonl(path, limit=None, balanced=True, seed=42):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if limit and len(rows) > limit:
        rnd = random.Random(seed)
        if balanced:  # 상황별로 고르게
            by = {}
            for r in rows:
                by.setdefault(r["scenario"], []).append(r)
            for v in by.values():
                rnd.shuffle(v)
            out, i = [], 0
            keys = sorted(by)
            while len(out) < limit:
                progressed = False
                for k in keys:
                    if i < len(by[k]):
                        out.append(by[k][i])
                        progressed = True
                        if len(out) >= limit:
                            break
                if not progressed:
                    break
                i += 1
            rows = out
        else:
            rnd.shuffle(rows)
            rows = rows[:limit]
    return rows
