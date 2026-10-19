# -*- coding: utf-8 -*-
"""
처방 LLM 학습 데이터 - 팀원 전처리 기준

기준을 팀원 전처리에 맞춘다.
  - 사례 = 팀원 코호트(cohort_final.parquet)의 ICU 입실 1건, 시점은 time zero (입실 48시간 후) 한 번
  - 분할 = 팀원 splits.parquet (split_random: 환자 단위 70/15/15)
  - 위험 점수 = train_risk_team.py 가 만든 _work/team/risk_scores.parquet
    (AKI 확률(보정), 신호등, 주요 요인 3개. train 은 교차 예측 점수)
  - 정답 = time zero 이후 --window-hours (기본 24시간, 팀원의 예측 구간과 같음) 안의 AKI 관련 약물 처방 변경
    (신규처방·감량·증량·중단. 정의는 prep_aki.py 의 rx_samples 와 같음)
  - 프롬프트의 환자 상태 요약(검사·활력징후·투여 약·신장/수분 줄)은 기존 추출 결과(_work/*.parquet)에서 time zero 이전
    값만 가져온다. 같은 MIMIC 원자료이고, 팀원 파이프라인은 약 처방 변경 정보를 만들지 않기 때문이다.

이전 방식과 다른 점: 입원 중 여러 시점·AKI 감지 시점 사례는 쓰지 않는다 (팀원 코호트는 time zero 이전 AKI 입실을 제외).
기존 data/ 폴더는 건드리지 않고 data_team/ 에 쓴다.

출력: data_team/{train,val,test}.jsonl, results/summary_rx_team.json (집계만, 10 미만은 <10)
사용:  python prep_rx_team.py [--window-hours 24] [--green-ratio -1] [--min-drug-count 15] [--no-kidney]
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as C
from prep_aki import H, load_cohort, rx_samples
from prep_build import log
from prep_rx import risk_header

SPLITS = ("train", "val", "test")


def _f(v, fmt):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else fmt % v


def trend_text(r):
    """trend_features.parquet 한 줄 -> 프롬프트용 [최근 변화] 문장 (값이 없는 항목은 뺀다)"""
    parts = []
    s = _f(r.get("tr_cr_delta_24h"), "%+.2f")
    if s:
        parts.append("크레아티닌 최근 24시간 %s" % s)
    s = _f(r.get("tr_cr_rise_from_min"), "%+.2f")
    if s:
        parts.append("48시간 최저 대비 %s" % s)
    s = _f(r.get("tr_cr_ratio_baseline"), "%.2f")
    if s:
        parts.append("기저치의 %s배" % s)
    lo, run, n12 = r.get("tr_uo_low_h_12h"), r.get("tr_uo_low_run_end"), r.get("tr_uo_obs_12h")
    if _f(n12, "%d") and float(n12) > 0 and _f(lo, "%d"):
        parts.append("최근 12시간 소변 0.5 mL/kg/h 미만 %d시간 (현재 연속 %s시간)" % (int(lo), _f(run, "%d") or "?"))
    s = _f(r.get("tr_uo_rate_6h"), "%.2f")
    if s:
        parts.append("최근 6시간 소변량 %s mL/kg/h" % s)
    s = _f(r.get("tr_mbp_min_12h"), "%.0f")
    if s:
        parts.append("최근 12시간 평균동맥압 최저 %s" % s)
    s = _f(r.get("tr_mbp_lt65_h_24h"), "%d")
    if s and int(s) > 0:
        parts.append("최근 24시간 평균동맥압 65 미만 %s시간" % s)
    s = _f(r.get("tr_fluid_bal_24h"), "%+.0f")
    if s:
        parts.append("최근 24시간 수분 균형 %s mL" % s)
    s = _f(r.get("tr_vaso_h_12h"), "%d")
    if s and int(s) > 0:
        parts.append("최근 12시간 승압제 %s시간" % s)
    return ("[최근 변화] " + ", ".join(parts)) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(C.WORK, "team"))
    ap.add_argument("--out-dir", default=os.path.join(C.HERE, "data_team"))
    ap.add_argument("--split", default="split_random", choices=["split_random", "split_temporal", "split_hybrid"],
                    help="위험 모델(train_risk_team.py)에서 쓴 것과 같아야 한다")
    ap.add_argument("--window-hours", type=int, default=24, help="정답으로 보는 처방 변경 구간 (time zero 이후)")
    ap.add_argument("--green-ratio", type=float, default=-1,
                    help="-1 = 모든 사례 사용. 0 이상이면 초록 사례를 빨강·노랑 수의 이 배수만 (이전 방식은 1.0)")
    ap.add_argument("--min-drug-count", type=int, default=15, help="train 정답에서 이 횟수 미만인 약은 제외")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-kidney", action="store_true")
    ap.add_argument("--risk-scores", default="risk_scores.parquet",
                    help="team-dir 안의 위험 점수 파일 (추세 모델: risk_scores_trend.parquet)")
    ap.add_argument("--trend-line", action="store_true",
                    help="team-dir/trend_features.parquet 로 프롬프트에 [최근 변화] 줄을 더한다")
    a = ap.parse_args()
    t0 = time.time()
    C.WINDOW_HOURS = a.window_hours
    rng = np.random.RandomState(a.seed)
    td = a.team_dir
    for f in ("cohort_final.parquet", "splits.parquet", a.risk_scores) + (("trend_features.parquet",) if a.trend_line else ()):
        if not os.path.exists(os.path.join(td, f)):
            raise SystemExit("[오류] %s 이 없습니다. wsl_team_pipeline.sh -> train_risk_team.py 를 먼저 실행하세요." % os.path.join(td, f))

    labs, vitals, rx, st, cr = load_cohort()
    coh = pd.read_parquet(os.path.join(td, "cohort_final.parquet"),
                          columns=["stay_id", "hadm_id", "subject_id", "intime", "outtime", "time_zero"])
    coh["time_zero"] = pd.to_datetime(coh["time_zero"])
    sc = pd.read_parquet(os.path.join(td, a.risk_scores), columns=["stay_id", "split", "prob", "tier", "factors"])
    log("위험 점수: %s" % a.risk_scores)
    n_coh = len(coh)
    log("팀원 코호트 %s건" % f"{n_coh:,}")

    # 내 체류 표(icu_stays.parquet)의 입실 시각은 chartevents 로 추정한 값이라 정확하지 않다.
    # 입실·퇴실 시각, hadm_id, subject_id 는 팀원 코호트(MIMIC icustays 원본)의 값을 쓰고,
    # 내 체류 표에서는 나이·성별 등 나머지 열만 가져온다.
    own = st.drop(columns=["split", "hadm_id", "subject_id", "intime", "outtime"], errors="ignore")
    own = own.drop_duplicates("stay_id")
    ev = coh.merge(own, on="stay_id", how="left", indicator=True)
    miss = int((ev["_merge"] == "left_only").sum())
    ev = ev[ev["_merge"] == "both"].drop(columns="_merge")
    for c in ("stay_id", "hadm_id", "subject_id"):
        ev[c] = ev[c].astype("int64")
    ev["intime"] = pd.to_datetime(ev["intime"]).astype(st["intime"].dtype)
    ev["outtime"] = pd.to_datetime(ev["outtime"]).astype(st["intime"].dtype)
    ev["t"] = ev["time_zero"].astype(st["intime"].dtype)   # 다른 표(검사·처방)의 시각과 같은 시간 단위로
    if "T" in ev.columns:
        ev["T"] = ev["t"]
    dt = (ev["t"] - (ev["intime"] + 48 * H)).abs()
    log("내 체류 표와 연결 실패 %d건, time zero 가 입실+48시간과 다른 건 %d건" % (miss, int((dt > pd.Timedelta(minutes=1)).sum())))
    ok = ev["t"] + a.window_hours * H <= ev["dischtime"]
    log("time zero + %d시간 전에 퇴원해 정답 구간을 끝까지 볼 수 없는 건 %d건 제외" % (a.window_hours, int((~ok).sum())))
    ev = ev[ok]
    ev = ev.merge(sc, on="stay_id", how="inner")
    ev["split"] = ev["split"].astype(str)
    cov = {"cohort": n_coh, "no_stay_match": miss, "left_before_window_end": int((~ok).sum()), "used": len(ev)}

    if a.green_ratio >= 0:
        picked = []
        for sp in SPLITS:
            g = ev[ev["split"] == sp]
            hot, green = g[g["tier"] != "green"], g[g["tier"] == "green"]
            n_green = min(len(green), int(round(max(len(hot), 1) * a.green_ratio)))
            picked.append(pd.concat([hot, green.iloc[rng.permutation(len(green))[:n_green]]]))
        ev = pd.concat(picked)
    ev = ev.reset_index(drop=True)
    log("처방 사례 %s건 (train %s / val %s / test %s)" % (
        f"{len(ev):,}", *[f"{int((ev['split'] == s).sum()):,}" for s in SPLITS]))

    kid = None
    if not a.no_kidney and os.path.exists(os.path.join(C.WORK, "kidney_output.parquet")):
        from prep_kidney import Kidney
        kid = Kidney(ev["stay_id"].values, ev["hadm_id"].values)
        log("신장·수분 상태 정보를 프롬프트에 넣습니다")
    elif not a.no_kidney:
        log("[참고] _work/kidney_*.parquet 이 없어 신장·수분 줄 없이 만듭니다 (python prep_kidney.py)")

    e, prompts, labels, actives = rx_samples(ev, risk_header, C.RISK_ASK, cr, labs, vitals, rx, kid)
    n_trend = 0
    if a.trend_line:
        tf = pd.read_parquet(os.path.join(td, "trend_features.parquet")).drop_duplicates("stay_id").set_index("stay_id")
        for i, sid in enumerate(e["stay_id"].values):
            if sid in tf.index:
                line = trend_text(tf.loc[sid].to_dict())
                if line:
                    head, ask = prompts[i].rsplit("\n", 1)
                    prompts[i] = head + "\n" + line + "\n" + ask
                    n_trend += 1
        log("[최근 변화] 줄 추가 %s건 / %s건" % (f"{n_trend:,}", f"{len(prompts):,}"))

    cnt = Counter(d for lab, sp in zip(labels, e["split"]) if sp == "train" for k in C.ACTIONS for d in lab[k])
    vocab = {d for d, n in cnt.items() if n >= a.min_drug_count}
    labels = [{k: [d for d in L[k] if d in vocab][:C.MAX_LABEL_MEDS] for k in C.ACTIONS} for L in labels]

    os.makedirs(a.out_dir, exist_ok=True)
    files = {sp: open(os.path.join(a.out_dir, "%s.jsonl" % sp), "w", encoding="utf-8") for sp in SPLITS}
    kr = C.SCENARIO_KR["aki_risk"]
    for i, sp in enumerate(e["split"].values):
        ans = {"상황": kr}
        ans.update(labels[i])
        rec = {"scenario": "aki_risk", "label": labels[i], "active": actives[i],
               "messages": [{"role": "system", "content": C.SYSTEM_PROMPT},
                            {"role": "user", "content": prompts[i]},
                            {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}],
               "case_id": "aki_risk-%d-%s" % (int(e["hadm_id"].values[i]),
                                              pd.Timestamp(e["t"].values[i]).strftime("%Y%m%d%H%M")),
               "tier": str(e["tier"].values[i]), "prob": round(float(e["prob"].values[i]), 4),
               "factors": str(e["factors"].values[i])}
        files[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()

    summ = {"design": "팀원 코호트, 입실 48시간 한 시점, 정답은 이후 %d시간 안 AKI 관련 약물 처방 변경" % a.window_hours,
            "risk_scores": a.risk_scores, "trend_line_cases": n_trend,
            "split_version": a.split, "coverage": {k: C.suppress(v) for k, v in cov.items()},
            "samples": {sp: C.suppress((e["split"] == sp).sum()) for sp in SPLITS},
            "tier_share": {sp: {t: round(float((e.loc[e["split"] == sp, "tier"] == t).mean()), 3)
                                for t in ("red", "yellow", "green")} for sp in SPLITS},
            "vocab_size": len(vocab),
            "action_rate": {k: round(float(np.mean([len(L[k]) > 0 for L in labels])), 3) for k in C.ACTIONS},
            "empty_label_rate": round(float(np.mean([sum(len(v) for v in L.values()) == 0 for L in labels])), 3),
            "active_aki_drug_rate": round(float(np.mean([len(x) > 0 for x in actives])), 3),
            "action_rate_by_tier": {
                t: round(float(np.mean([sum(len(v) for v in L.values()) > 0
                                        for L, tt in zip(labels, e["tier"].values) if tt == t])), 3)
                for t in ("red", "yellow", "green") if (e["tier"] == t).sum() >= 10},
            "prompt_chars_mean": int(np.mean([len(p) for p in prompts])),
            "top_drugs_train": {},
            "kidney_info": None if kid is None else (
                {k: round(float(np.mean([c[k] for c in kid.coverage])), 3)
                 for k in ("weight", "egfr", "urine", "intake", "dialysis")} if kid.coverage else {})}
    for k in C.ACTIONS:
        c = Counter(d for L, sp in zip(labels, e["split"]) if sp == "train" for d in L[k])
        summ["top_drugs_train"][k] = [[d, C.suppress(n)] for d, n in c.most_common(10)]
    summ["minutes"] = round((time.time() - t0) / 60, 1)
    base = os.path.basename(os.path.normpath(a.out_dir))
    C.save_json(summ, os.path.join(C.RESULTS, "summary_rx_team.json" if base == "data_team" else "summary_rx_%s.json" % base))
    log("완료 (%.1f분). %s/{train,val,test}.jsonl, 집계: results/summary_rx_team.json" % (summ["minutes"], a.out_dir))
    log("다음(WSL):  bash \"<이 폴더>/wsl_train_team.sh\"   (학습 데이터 폴더 data_team 사용)")


if __name__ == "__main__":
    main()
