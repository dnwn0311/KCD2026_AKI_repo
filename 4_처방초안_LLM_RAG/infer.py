# -*- coding: utf-8 -*-
"""
처방 초안 생성 (검증 AI 에 넘길 파일 만들기)

data/<split>.jsonl 의 사례마다 Qwen3.5-9B + LoRA 어댑터로 AKI 관련 약물 처방 변경 초안을 만들고,
검증 AI(RAG)가 바로 읽을 수 있게 한 줄에 한 사례씩 저장한다.

출력: _work/drafts_<split>.jsonl  (환자 단위, 연구실 서버 밖으로 내보내지 않음)
  case_id   사례 번호 (상황-입원번호-시각)
  scenario  "aki_risk" (위험 예측 시점) 또는 "aki" (AKI 감지 시점)
  risk      위험 예측 시점일 때 {"prob": 확률, "tier": "red|yellow|green", "factors": 주요 요인}
  system    시스템 프롬프트,  prompt  환자 상태 요약 (LLM 입력)
  draft     초안 {"신규처방": [...], "감량": [...], "증량": [...], "중단": [...]}, 해석 실패 시 null
  raw       모델 원문 출력
  actual    실제 기록된 처방 변경 (정답, 평가용)
results/drafts_<split>_metrics.json 에는 실제 처방과의 일치도 집계만 저장

사용:  python infer.py [--split test] [--adapter runs/qlora_aki/adapter] [--limit 0]
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
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = 전체")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if not os.path.exists(os.path.join(a.adapter, "adapter_config.json")):
        raise SystemExit("[오류] 어댑터를 찾지 못했습니다: %s" % a.adapter)

    dtype = L.pick_dtype()
    tok = L.load_tokenizer(a.adapter if os.path.exists(os.path.join(a.adapter, "tokenizer_config.json")) else a.model)
    tok.padding_side = "left"
    model = L.load_model(a.model, not a.no_4bit, dtype)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    if a.no_4bit and torch.cuda.is_available():
        model.to("cuda")
    device = next(model.parameters()).device

    with open(os.path.join(C.DATA, "%s.jsonl" % a.split), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if a.limit:
        rows = rows[:a.limit]
    stops = L.end_token_ids(tok)
    os.makedirs(C.WORK, exist_ok=True)
    out_path = os.path.join(C.WORK, "drafts_%s.jsonl" % a.split)
    results = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            prompts = [L.render_prompt(tok, r["messages"][:-1]) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     eos_token_id=stops or None, pad_token_id=tok.pad_token_id)
            gen = gen[:, enc["input_ids"].shape[1]:]
            for r, g in zip(chunk, gen):
                text = tok.decode(g, skip_special_tokens=True)
                draft = C.parse_answer(text)
                rec = {"case_id": r.get("case_id"), "scenario": r["scenario"],
                       "risk": ({"prob": r.get("prob"), "tier": r.get("tier"), "factors": r.get("factors")}
                                if r["scenario"] == "aki_risk" else None),
                       "system": r["messages"][0]["content"], "prompt": r["messages"][1]["content"],
                       "draft": draft, "raw": text, "actual": r["label"]}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append((r["scenario"], r["label"], draft))
            print("  %d/%d" % (min(i + a.batch, len(rows)), len(rows)), flush=True)
    res = C.set_metrics(results)
    res.update({"split": a.split, "adapter": a.adapter, "model": a.model, "minutes": round((time.time() - t0) / 60, 1)})
    C.save_json(res, os.path.join(C.RESULTS, "drafts_%s_metrics.json" % a.split))
    o = res["overall"]
    print("[완료] %s (%d건). JSON 해석 %.1f%%, 실제 처방과의 micro F1 %.3f"
          % (out_path, len(results), 100 * (o["parse_rate"] or 0), o["micro_f1"]))


if __name__ == "__main__":
    main()
