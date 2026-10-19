# -*- coding: utf-8 -*-
"""
처방 LLM 학습 데이터 (위험 예측 모델을 학습한 뒤 실행)

흐름: 위험 예측 → LLM 이 예측 결과를 보고 AKI 관련 약물 처방 변경 초안 → 의사 확인
두 종류의 사례를 함께 학습한다.

1. 위험 예측 시점 (scenario = "aki_risk")
   - 이동 창 예측 시점마다 (아직 AKI 없음). 입력에 위험 예측 결과를 넣는다:
     향후 24시간 AKI 확률(보정), 신호등 색, 위험을 올린 주요 요인 3개 (train 은 교차 예측 점수)
   - 정답: 예측 시점 이후 12시간 안의 AKI 관련 약물 처방 변경 (예방적 조정)
   - 표본: 빨강·노랑 시점은 모두, 초록 시점은 같은 수만큼 무작위로 (입력 값으로만 고르므로 정답 분포를 건드리지 않음)
2. AKI 감지 시점 (scenario = "aki")
   - 입실 48시간 이후 중환자실에서 AKI 가 감지된 시점. 정답: 감지 후 12시간 안의 AKI 관련 약물 처방 변경

정답 정의 (신규처방·감량·증량·중단) 는 prep_build.py·prep_aki.py 와 같다.
출력: data/{train,val,test}.jsonl, results/summary_rx.json (집계만, 10 미만은 <10)

신장·수분 상태: _work/kidney_*.parquet (prep_kidney.py) 가 있으면 프롬프트에 [신장·수분 상태] 줄을 넣는다.
  사례 선택과 정답은 이 줄이 있든 없든 같다 (같은 seed 면 같은 사례 -> 이전 결과와 짝지어 비교 가능)

사용:  python prep_rx.py [--green-ratio 1.0] [--cap-risk-train 8000] [--cap-aki-train 6000] [--min-drug-count 20]
                        [--no-kidney]
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as C
from prep_aki import H, aki_events, aki_header, load_cohort, rx_samples
from prep_build import log

SPLITS = ("train", "val", "test")


def risk_points(st, rng, green_ratio, caps):
    risk = pd.read_parquet(os.path.join(C.DATA, "aki_risk.parquet"), columns=["pid", "stay_key", "icu_hours_at_T"])
    p = os.path.join(C.WORK, "risk_scores.parquet")
    if not os.path.exists(p):
        raise SystemExit("[오류] _work/risk_scores.parquet 이 없습니다. train_risk.py 를 먼저 실행하세요.")
    sc = pd.read_parquet(p)
    pts = risk.merge(sc, on="pid", how="inner").rename(columns={"stay_key": "hadm_id"})
    pts = pts.merge(st, on="hadm_id", how="inner", suffixes=("", "_st"))
    pts["split"] = pts["split"].astype(str)
    pts["t"] = pts["intime"] + pts["icu_hours_at_T"].astype(float) * H
    pts = pts[pts["t"] + C.WINDOW_HOURS * H <= pts["dischtime"]]
    picked, stat = [], {}
    for sp in SPLITS:
        g = pts[pts["split"] == sp]
        hot = g[g["tier"] != "green"]
        green = g[g["tier"] == "green"]
        n_green = min(len(green), int(round(max(len(hot), 1) * green_ratio)))
        s = pd.concat([hot, green.iloc[rng.permutation(len(green))[:n_green]]])
        if len(s) > caps[sp]:
            s = s.iloc[rng.permutation(len(s))[:caps[sp]]]
        picked.append(s)
        stat[sp] = {t: C.suppress((s["tier"] == t).sum()) for t in ("red", "yellow", "green")}
    return pd.concat(picked, ignore_index=True), stat


def risk_header(r):
    lines = ["[AKI 위험 예측] 향후 24시간 안 AKI 발생 확률 %.0f%%, 신호등 %s"
             % (100 * float(r["prob"]), C.TIER_KR.get(r["tier"], r["tier"]))]
    if isinstance(r["factors"], str) and r["factors"]:
        lines.append("[위험을 올린 주요 요인] " + r["factors"])
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--green-ratio", type=float, default=1.0, help="초록 시점을 빨강·노랑 수의 몇 배만큼 넣을지")
    ap.add_argument("--cap-risk-train", type=int, default=8000)
    ap.add_argument("--cap-risk-val", type=int, default=400)
    ap.add_argument("--cap-risk-test", type=int, default=800)
    ap.add_argument("--cap-aki-train", type=int, default=6000)
    ap.add_argument("--cap-aki-val", type=int, default=300)
    ap.add_argument("--cap-aki-test", type=int, default=600)
    ap.add_argument("--min-drug-count", type=int, default=20, help="train 정답에서 이 횟수 미만인 약은 제외")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-kidney", action="store_true", help="[신장·수분 상태] 줄을 넣지 않음 (이전 버전과 같은 입력)")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(a.seed)
    labs, vitals, rx, st, cr = load_cohort()

    pts, tier_stat = risk_points(st, rng, a.green_ratio,
                                 {"train": a.cap_risk_train, "val": a.cap_risk_val, "test": a.cap_risk_test})
    log("위험 예측 시점 사례 %s건" % f"{len(pts):,}")
    ev = aki_events(st, cr)
    caps = {"train": a.cap_aki_train, "val": a.cap_aki_val, "test": a.cap_aki_test}
    ev = pd.concat([g.iloc[rng.permutation(len(g))[:caps[sp]]] for sp, g in ev.groupby("split")], ignore_index=True)
    log("AKI 감지 사례 %s건" % f"{len(ev):,}")

    kid = None
    if not a.no_kidney and os.path.exists(os.path.join(C.WORK, "kidney_output.parquet")):
        from prep_kidney import Kidney
        ids = pd.concat([pts[["stay_id", "hadm_id"]], ev[["stay_id", "hadm_id"]]])
        kid = Kidney(ids["stay_id"].values, ids["hadm_id"].values)
        log("신장·수분 상태 정보를 프롬프트에 넣습니다 (_work/kidney_*.parquet)")
    elif not a.no_kidney:
        log("[참고] _work/kidney_*.parquet 이 없어 신장·수분 상태 없이 만듭니다 (python prep_kidney.py 로 추출)")

    groups = []
    for scen, df, hfn, ask in (("aki_risk", pts, risk_header, C.RISK_ASK), ("aki", ev, aki_header, C.AKI_ASK)):
        if len(df) == 0:
            continue
        e, prompts, labels, actives = rx_samples(df, hfn, ask, cr, labs, vitals, rx, kid)
        groups.append((scen, e, prompts, labels, actives))

    # 정답 약물 사전: train 정답(두 종류 합쳐서)에서 충분히 나온 약만
    cnt = Counter(d for scen, e, _, L, _ in groups for lab, sp in zip(L, e["split"]) if sp == "train"
                  for k in C.ACTIONS for d in lab[k])
    vocab = {d for d, n in cnt.items() if n >= a.min_drug_count}

    os.makedirs(C.DATA, exist_ok=True)
    files = {sp: open(os.path.join(C.DATA, "%s.jsonl" % sp), "w", encoding="utf-8") for sp in SPLITS}
    summ = {"design": "위험 예측 시점(예방적 조정)과 AKI 감지 시점(대응) 두 종류, 정답은 이후 %d시간 안 AKI 관련 약물 변경"
                      % C.WINDOW_HOURS,
            "vocab_size": len(vocab), "risk_point_tiers": tier_stat, "by_scenario": {},
            "kidney_info": None if kid is None else {
                k: round(float(np.mean([c[k] for c in kid.coverage])), 3)
                for k in ("weight", "egfr", "urine", "intake", "dialysis")} if kid.coverage else {}}
    for scen, e, prompts, labels, actives in groups:
        labels = [{k: [d for d in L[k] if d in vocab][:C.MAX_LABEL_MEDS] for k in C.ACTIONS} for L in labels]
        kr = C.SCENARIO_KR[scen]
        for i, sp in enumerate(e["split"].values):
            ans = {"상황": kr}
            ans.update(labels[i])
            rec = {"scenario": scen, "label": labels[i], "active": actives[i],
                   "messages": [{"role": "system", "content": C.SYSTEM_PROMPT},
                                {"role": "user", "content": prompts[i]},
                                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]}
            rec["case_id"] = "%s-%d-%s" % (scen, int(e["hadm_id"].values[i]),
                                            pd.Timestamp(e["t"].values[i]).strftime("%Y%m%d%H%M"))
            if scen == "aki_risk":
                rec["tier"] = str(e["tier"].values[i])
                rec["prob"] = round(float(e["prob"].values[i]), 4)
                rec["factors"] = str(e["factors"].values[i])
            files[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
        s = {"samples": {sp: C.suppress((e["split"] == sp).sum()) for sp in SPLITS},
             "action_rate": {k: round(float(np.mean([len(L[k]) > 0 for L in labels])), 3) for k in C.ACTIONS},
             "empty_label_rate": round(float(np.mean([sum(len(v) for v in L.values()) == 0 for L in labels])), 3),
             "active_aki_drug_rate": round(float(np.mean([len(x) > 0 for x in actives])), 3),
             "prompt_chars_mean": int(np.mean([len(p) for p in prompts])), "top_drugs_train": {}}
        if scen == "aki_risk":
            s["action_rate_by_tier"] = {
                t: round(float(np.mean([sum(len(v) for v in L.values()) > 0
                                        for L, tt in zip(labels, e["tier"].values) if tt == t])), 3)
                for t in ("red", "yellow", "green") if (e["tier"] == t).sum() >= 10}
        for k in C.ACTIONS:
            c = Counter(d for L, sp in zip(labels, e["split"]) if sp == "train" for d in L[k])
            s["top_drugs_train"][k] = [[d, C.suppress(n)] for d, n in c.most_common(10)]
        summ["by_scenario"][scen] = s
    for f in files.values():
        f.close()
    summ["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(summ, os.path.join(C.RESULTS, "summary_rx.json"))
    log("처방 LLM 데이터 완료 (%.1f분). data/{train,val,test}.jsonl, 집계: results/summary_rx.json" % summ["minutes"])


if __name__ == "__main__":
    main()
