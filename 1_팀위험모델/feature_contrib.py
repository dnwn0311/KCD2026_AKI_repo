# -*- coding: utf-8 -*-
"""
위험 예측 모델(runs/aki_risk_team_trend)의 변수 기여도를 전체 280개에 대해 계산한다. (집계만 저장, 환자 단위 행은 저장·출력하지 않음)

  - mean|SHAP|  : 테스트 1,658건 전체에서 변수별 평균 절대 기여도 (LightGBM pred_contrib, 로그오즈 단위)
  - gain / split: 모델 파일 안의 분할 이득 합과 분할 횟수 (SHAP 없이도 계산되는 보조 지표)
  - 방향        : 변수 값이 높은(상위 25%) 환자와 낮은(하위 25%) 환자의 평균 SHAP, 값과 SHAP의 순위상관 (수치형만)
  - 계열별 합   : 약 10군 / 약 ATC3 / 추세 / 소변량·수액 / 활력 / 검사 / 정적

실행 (WSL, 팀원 전처리 산출물이 있는 _work/team_fixed 기준):
  source ~/kcd_venv/bin/activate
  cd /mnt/c/Users/User/Documents/KCD_처방추천
  python feature_contrib.py
옵션: --team-dir <폴더>  --split split_random  --model runs/aki_risk_team_trend/model.txt  --out feature_contrib
결과: <out>/feature_contrib_all.csv, feature_contrib_groups.csv, feature_contrib_top30.txt
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FORBIDDEN_PREFIX = ("label_", "aki_by_", "onset_", "n_outlier_", "split_")
VITALS = ("mbp", "sbp", "dbp", "hr", "rr", "temp_c")
LABS = ("bun", "cl", "creatinine", "hco3", "k", "mg", "na", "phosphate", "urine_creat")
UF = ("urine_ml", "urine_ml_kg", "intake", "drain", "stool", "other_output", "fluid_balance")


def group_of(n):
    if n.startswith("tr_"):
        return "추세(tr_)"
    if n.startswith("atc3_"):
        return "약 계열 ATC3 플래그(124)"
    if n.startswith("atc_") or n == "drug_pip_tazo":
        return "신독성 약 10군"
    if n == "n_kidney_relevant_classes":
        return "신독성 약 10군"
    if n.startswith("egfr_") or n in ("creat_baseline_7d", "creat_last_pre_icu") or any(n.startswith(p + "_") for p in LABS):
        return "검사(크레아티닌·eGFR·전해질 등)"
    if any(n.startswith(p + "_") for p in VITALS):
        return "활력징후"
    if n.startswith(UF) or n.startswith("weight_daily") or n == "urine_ml_kg_h_overall":
        return "소변량·수액·체중"
    return "정적·치료력(나이, 병동, 투석, 승압제 등)"


def parse_model(model_path):
    txt = open(model_path, encoding="utf-8").read()
    names = re.search(r"^feature_names=(.*)$", txt, re.M).group(1).split(" ")
    gain, split = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0)
    for blk in txt.split("Tree=")[1:]:
        sf = re.search(r"^split_feature=(.*)$", blk, re.M)
        sg = re.search(r"^split_gain=(.*)$", blk, re.M)
        if not sf or not sf.group(1).strip():
            continue
        for f, g in zip(sf.group(1).split(), sg.group(1).split()):
            gain[names[int(f)]] += float(g)
            split[names[int(f)]] += 1
    return names, gain, split


def main():
    import lightgbm as lgb
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(HERE, "_work", "team_fixed"))
    ap.add_argument("--split", default="split_random")
    ap.add_argument("--model", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "model.txt"))
    ap.add_argument("--config", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "feature_contrib"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = json.load(open(a.config, encoding="utf-8"))
    feats, cats = cfg["features"], cfg["categorical"]
    dfeat = json.load(open(os.path.join(a.team_dir, "default_features.json"), encoding="utf-8"))
    label_col = dfeat["label"]

    wide = pd.read_parquet(os.path.join(a.team_dir, "wide.parquet"))
    tf = pd.read_parquet(os.path.join(a.team_dir, "trend_features.parquet"))
    spl = pd.read_parquet(os.path.join(a.team_dir, "splits.parquet"), columns=["stay_id", a.split])
    df = wide.merge(spl, on="stay_id", how="inner").merge(tf, on="stay_id", how="left")
    df = df.rename(columns={a.split: "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    for c in cats:   # 학습 때와 같은 범주 순서 (train 에 나온 값, 정렬)
        vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    for c in feats:
        if c in cats:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype("int8")
        elif df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    te = df[df["split"] == "test"]
    X = te[feats]
    y = te["label"].astype(int).values
    print("테스트 %d건, 양성 %d건, 특징 %d개" % (len(te), y.sum(), len(feats)))

    booster = lgb.Booster(model_file=a.model)
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    # 확인: 모델 원점수 = 기여도 합 + 기준값
    raw = booster.predict(X, raw_score=True)
    full = booster.predict(X, pred_contrib=True)
    print("SHAP 합 검증(최대 오차): %.2e" % np.abs(full.sum(axis=1) - raw).max())

    names, gain, split = parse_model(a.model)
    tot_gain = sum(gain.values()) or 1.0
    mabs = np.abs(contrib).mean(axis=0)
    rows = []
    for j, n in enumerate(feats):
        row = {"feature": n, "group": group_of(n), "mean_abs_shap": float(mabs[j]),
               "gain_share_pct": 100 * gain.get(n, 0.0) / tot_gain, "n_splits": split.get(n, 0),
               "mean_shap_on_positives": float(contrib[y == 1, j].mean()),
               "mean_shap_on_negatives": float(contrib[y == 0, j].mean())}
        v = X[n]
        if n not in cats and pd.api.types.is_numeric_dtype(v) and v.notna().sum() > 30 and v.nunique() > 2:
            vv = v.values.astype(float)
            ok = ~np.isnan(vv)
            q1, q3 = np.nanpercentile(vv, [25, 75])
            hi, lo = ok & (vv >= q3), ok & (vv <= q1)
            row["mean_shap_high25"] = float(contrib[hi, j].mean()) if hi.sum() else np.nan
            row["mean_shap_low25"] = float(contrib[lo, j].mean()) if lo.sum() else np.nan
            row["spearman_value_vs_shap"] = float(spearmanr(vv[ok], contrib[ok, j]).correlation)
            row["missing_pct"] = float(100 * (~ok).mean())
        else:
            row["missing_pct"] = float(100 * v.isna().mean())
        rows.append(row)
    R = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    R["shap_share_pct"] = 100 * R["mean_abs_shap"] / R["mean_abs_shap"].sum()
    R.insert(0, "rank", range(1, len(R) + 1))
    R.round(5).to_csv(os.path.join(a.out, "feature_contrib_all.csv"), index=False, encoding="utf-8-sig")

    G = R.groupby("group").agg(n_features=("feature", "count"), shap_share_pct=("shap_share_pct", "sum"),
                               gain_share_pct=("gain_share_pct", "sum"),
                               n_used=("n_splits", lambda s: int((s > 0).sum()))).sort_values("shap_share_pct", ascending=False)
    G.round(2).to_csv(os.path.join(a.out, "feature_contrib_groups.csv"), encoding="utf-8-sig")

    drug = R[R["group"] == "신독성 약 10군"][["feature", "mean_abs_shap", "shap_share_pct", "n_splits"]]
    lines = ["[계열별 기여 비중]", G.round(2).to_string(), "", "[상위 30 변수]"]
    for _, r in R.head(30).iterrows():
        d = ""
        if "spearman_value_vs_shap" in r and pd.notna(r.get("spearman_value_vs_shap")):
            d = "  값↑ → 위험 %s (상관 %+.2f)" % ("↑" if r["spearman_value_vs_shap"] > 0 else "↓", r["spearman_value_vs_shap"])
        lines.append("%2d. %-34s |SHAP| %.4f (%.1f%%)  gain %.1f%%  [%s]%s" % (
            r["rank"], r["feature"], r["mean_abs_shap"], r["shap_share_pct"], r["gain_share_pct"], r["group"], d))
    lines += ["", "[신독성 약 10군 + 개수 변수]", drug.round(4).to_string(index=False),
              "", "모델이 한 번도 쓰지 않은 변수: %d개 (그중 ATC3 %d개)" % (
                  int((R["n_splits"] == 0).sum()), int(((R["n_splits"] == 0) & R["feature"].str.startswith("atc3_")).sum()))]
    out = "\n".join(lines)
    open(os.path.join(a.out, "feature_contrib_top30.txt"), "w", encoding="utf-8").write(out)
    print(out)
    print("\n저장:", a.out)


if __name__ == "__main__":
    main()
