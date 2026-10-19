# -*- coding: utf-8 -*-
"""
받은 위험 예측 모델로 점수 매기기 (다시 학습하지 않음)

전달받은 runs/aki_risk/ (model.txt, config.json) 로 data/aki_risk.parquet 의 모든 예측 시점에
보정 확률, 신호등 색, 위험을 올린 주요 요인을 붙여 _work/risk_scores.parquet 에 저장한다.
그다음 prep_rx.py 를 실행하면 처방 LLM 입력(JSONL)이 만들어진다.

- 보정·신호등 기준은 보낸 쪽이 val 로 정한 값을 그대로 쓴다 (config.json)
- train 분할의 점수는 이 모델이 학습한 데이터라 실제보다 좋게 나온다. 검증에는 val·test 만 쓰세요
- 보낸 쪽 test 성능(config.json 의 test_auroc)과 비교해 같은 데이터가 만들어졌는지 확인한다

사용:  python score_risk.py [--model-dir runs/aki_risk]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import common as C
from train_risk import explain_top, log, metrics, tier_of


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(C.HERE, "runs", "aki_risk"))
    a = ap.parse_args()
    with open(os.path.join(a.model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    # 한글이 든 Windows 경로에서도 열리도록 파이썬으로 읽어서 문자열로 넘긴다
    with open(os.path.join(a.model_dir, "model.txt"), encoding="utf-8") as f:
        booster = lgb.Booster(model_str=f.read())
    df = pd.read_parquet(os.path.join(C.DATA, "aki_risk.parquet"))
    feats = cfg["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit("[오류] 특징이 모델과 다릅니다 (코드 버전 확인): %s" % missing[:10])
    for c in ("admission_type", "admission_location"):
        if c in df:
            df[c] = df[c].astype("category")
    X = df[feats]
    raw = booster.predict(X)
    cal = np.clip(np.interp(raw, cfg["isotonic_x"], cfg["isotonic_y"]), 0, 1)
    tier = tier_of(cal, cfg["red_threshold"], cfg["yellow_threshold"])
    log("위험 요인 설명 계산 (%s개 시점)" % f"{len(df):,}")
    facs = explain_top(booster, X, feats)
    out = pd.DataFrame({"pid": df["pid"].values, "split": df["split"].values, "prob": cal.astype("float32"),
                        "tier": tier, "factors": facs})
    os.makedirs(C.WORK, exist_ok=True)
    out.to_parquet(os.path.join(C.WORK, "risk_scores.parquet"), index=False)

    rep = {"model_dir": a.model_dir, "sender_test_auroc": cfg.get("test_auroc")}
    for sp in ("val", "test"):
        m = df["split"].values == sp
        if m.any():
            rep[sp] = metrics(df.loc[m, "label"].values, raw[m])
            rep[sp]["tier_share"] = {t: round(float(v), 4) for t, v in
                                     pd.Series(tier[m]).value_counts(normalize=True).items()}
    C.save_json(rep, os.path.join(C.RESULTS, "score_risk_check.json"))
    mine, theirs = (rep.get("test") or {}).get("auroc"), cfg.get("test_auroc")
    log("test AUROC: 이 서버 %s / 보낸 쪽 %s" % (mine, theirs))
    if mine is not None and theirs is not None and abs(mine - theirs) > 0.005:
        log("[주의] test 성능이 보낸 쪽과 다릅니다. MIMIC 버전이나 코드 버전이 같은지 확인하세요.")
    log("완료: _work/risk_scores.parquet. 다음: python prep_rx.py")


if __name__ == "__main__":
    main()
