# -*- coding: utf-8 -*-
"""
확률 기반 평가: 사례마다 답을 여러 번 뽑아(샘플링) 약마다 "몇 번 나왔는지"로 처방 여부를 정한다

배경: 정답의 대부분이 "변경 없음"이라, 가장 그럴듯한 답 하나만 뽑는 방식(탐욕 디코딩)은 거의 항상 빈 답이 된다
(eval_model.py 결과). 같은 모델에서 답을 n 번 뽑으면 각 (변경 종류, 약)의 등장 비율이 그 모델의 확신 정도가 되고,
비율이 기준값(τ) 이상인 약만 내놓으면 정밀도와 재현율의 균형을 조절할 수 있다.

사용 (2단계)
  1) val 에서 변경 종류별 최적 τ 찾기:
       python eval_sample.py --split val --limit 200 --n 16 --tag samp_val
     -> results/tau_samp_val.json (변경 종류별 τ), 터미널에 τ별 F1 표
  2) 찾은 τ 를 test 에 적용:
       python eval_sample.py --split test --limit 600 --n 16 --tag samp_test --tau results/tau_samp_val.json
     -> results/eval_samp_test.json

출력: _work/votes_<tag>.jsonl (환자 단위, 서버 밖으로 내보내지 않음), results/ 아래 집계만
"""
import argparse
import json
import os
import time

import common as C


def thresholds(n):
    return [round(k / n, 4) for k in range(1, n + 1)]


def make_pred(votes, tau):
    """votes: {종류: {약: 비율}}, tau: 숫자 또는 {종류: 숫자}"""
    out = {}
    for k in C.ACTIONS:
        t = tau[k] if isinstance(tau, dict) else tau
        out[k] = sorted(d for d, f in votes.get(k, {}).items() if f >= t)
    return out


def score(cases, tau):
    res = [(c["scenario"], c["label"], make_pred(c["votes"], tau)) for c in cases]
    m = C.set_metrics(res)
    m["items_per_case"] = round(sum(sum(len(p[k]) for k in C.ACTIONS) for _, _, p in res) / max(len(res), 1), 3)
    return m


def f1_of(m, k=None):
    o = m["overall"]
    v = o["micro_f1"] if k is None else o["by_action"][k]["f1"]
    return v or 0.0


def sweep_table(cases, n):
    rows = []
    for t in thresholds(n):
        m = score(cases, t)
        rows.append((t, m))
    return rows


def best_tau_per_action(table):
    best = {}
    for k in C.ACTIONS:
        t_best, f_best = None, 0.0
        for t, m in table:
            f = f1_of(m, k)
            if f > f_best:
                t_best, f_best = t, f
        best[k] = {"tau": t_best if t_best is not None else 1.01, "val_f1": round(f_best, 4)}
    return best


def print_table(table):
    print("  τ     전체F1  신규   감량   증량   중단   사례당 항목")
    for t, m in table:
        o = m["overall"]
        print("  %.3f %6.3f %6.3f %6.3f %6.3f %6.3f  %5.2f" % (
            t, o["micro_f1"] or 0, *[(o["by_action"][k]["f1"] or 0) for k in C.ACTIONS], m["items_per_case"]),
            flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default=os.path.join(C.HERE, "runs", "qlora_aki", "adapter"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=200, help="0 = 전체")
    ap.add_argument("--n", type=int, default=16, help="사례마다 뽑는 답의 수")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2, help="한 번에 처리하는 사례 수 (실제 시퀀스 = batch x n)")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--tau", default=None, help="변경 종류별 τ 가 든 json (val 에서 구한 것). 없으면 τ 를 새로 찾는다")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()
    tag = a.tag or ("samp_" + a.split)

    import torch
    import llm_common as L
    from peft import PeftModel

    t0 = time.time()
    dtype = L.pick_dtype()
    tok = L.load_tokenizer(a.adapter)
    tok.padding_side = "left"
    model = L.load_model(a.model, not a.no_4bit, dtype)
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    if a.no_4bit and torch.cuda.is_available():
        model.to("cuda")
    device = next(model.parameters()).device

    rows = L.read_jsonl(os.path.join(C.DATA, "%s.jsonl" % a.split), a.limit or None)
    stops = L.end_token_ids(tok)
    os.makedirs(C.WORK, exist_ok=True)
    cases, fails = [], 0
    with open(os.path.join(C.WORK, "votes_%s.jsonl" % tag), "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            prompts = [L.render_prompt(tok, r["messages"][:-1]) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temp,
                                     top_p=1.0, top_k=0, num_return_sequences=a.n,
                                     eos_token_id=stops or None, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, r in enumerate(chunk):
                cnt = {k: {} for k in C.ACTIONS}
                for s in range(a.n):
                    text = tok.decode(gen[j * a.n + s], skip_special_tokens=True)
                    p = C.parse_answer(text)
                    if p is None:
                        fails += 1
                        continue
                    for k in C.ACTIONS:
                        for d in set(p[k]):
                            cnt[k][d] = cnt[k].get(d, 0) + 1
                votes = {k: {d: round(v / a.n, 4) for d, v in cnt[k].items()} for k in C.ACTIONS}
                c = {"case_id": r.get("case_id"), "scenario": r["scenario"], "label": r["label"], "votes": votes}
                cases.append(c)
                fo.write(json.dumps(c, ensure_ascii=False) + "\n")
            done = min(i + a.batch, len(rows))
            el = (time.time() - t0) / 60
            print("  %d/%d (%.1f분 경과, 남은 시간 약 %.0f분)" % (done, len(rows), el, el * (len(rows) - done) / max(done, 1)),
                  flush=True)

    print("[샘플링 완료] %d건, 해석 실패 %.1f%% (%.1f분)" % (len(cases), 100 * fails / (len(cases) * a.n),
                                                          (time.time() - t0) / 60))
    table = sweep_table(cases, a.n)
    print("\n== 모든 종류에 같은 τ 를 쓴 결과 (%s)" % a.split)
    print_table(table)

    if a.tau is None:
        best = best_tau_per_action(table)
        C.save_json({"tau": {k: v["tau"] for k, v in best.items()}, "detail": best, "split": a.split,
                     "n": a.n, "cases": len(cases)}, os.path.join(C.RESULTS, "tau_%s.json" % tag))
        print("\n== 종류별 최적 τ (이 split 기준): %s" % json.dumps(best, ensure_ascii=False))
        print("다음: python eval_sample.py --split test --limit 600 --tag samp_test --tau results/tau_%s.json" % tag)
        return

    with open(a.tau, encoding="utf-8") as f:
        tau = json.load(f)["tau"]
    m = score(cases, tau)
    m.update({"tag": tag, "split": a.split, "tau": tau, "n_samples": a.n, "temperature": a.temp,
              "json_fail_rate": round(fails / (len(cases) * a.n), 4),
              "minutes": round((time.time() - t0) / 60, 1)})
    C.save_json(m, os.path.join(C.RESULTS, "eval_%s.json" % tag))
    o = m["overall"]
    print("\n[%s] τ=%s -> 샘플 JSON 실패 %.1f%%, 전체 micro F1 %.3f (신규 %.3f, 감량 %.3f, 증량 %.3f, 중단 %.3f), 사례당 항목 %.2f"
          % (tag, json.dumps(tau, ensure_ascii=False), 100 * m["json_fail_rate"], o["micro_f1"] or 0,
             *[(o["by_action"][k]["f1"] or 0) for k in C.ACTIONS], m["items_per_case"]))


if __name__ == "__main__":
    main()
