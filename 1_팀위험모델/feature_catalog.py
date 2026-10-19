# -*- coding: utf-8 -*-
"""
위험 예측 모델의 입력 변수 280개 전체의 '값 목록'을 집계로 만든다 (환자 단위 행은 저장·출력하지 않음, 10건 미만 칸은 가림).
  변수명, 계열, 형식, 설명, 단위, 결측 비율, 분포 값(1·5·25·50·75·95·99%), 평균, 범주별 비율(범주형), 1 의 비율(0/1 변수),
  평균 |SHAP|, 순위, 모델 사용 여부
실행:  python3 feature_catalog.py        (feature_contrib.py 를 먼저 돌려 feature_contrib/feature_contrib_all.csv 가 있어야 함)
결과:  feature_contrib/feature_catalog.csv
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_CELL = 10
VITALS = ("mbp", "sbp", "dbp", "hr", "rr", "temp_c")
LABS = ("bun", "cl", "creatinine", "hco3", "k", "mg", "na", "phosphate", "urine_creat")
UF = ("urine_ml", "intake", "drain", "stool", "other_output", "fluid_balance")
TREND_KR = {
    "tr_cr_n": "48시간 크레아티닌 측정 수", "tr_cr_hsl": "마지막 크레아티닌 측정 후 경과 시간",
    "tr_cr_delta_24h": "크레아티닌 최근 24시간 변화", "tr_cr_rise_from_min": "크레아티닌 48시간 최저 대비 상승",
    "tr_cr_rise_vs_ref": "크레아티닌 기준값(48시간·입실 전 최저) 대비 상승", "tr_cr_ratio_min": "크레아티닌 48시간 최저 대비 비율",
    "tr_cr_ratio_baseline": "크레아티닌 기저치 대비 비율", "tr_cr_slope_day": "크레아티닌 하루 기울기",
    "tr_cr_max_rise": "48시간 안 크레아티닌 최대 상승폭",
    "tr_bun_delta_24h": "BUN 최근 24시간 변화", "tr_bun_cr_ratio": "BUN/크레아티닌 비",
    "tr_k_delta_24h": "칼륨 최근 24시간 변화", "tr_hco3_delta_24h": "중탄산 최근 24시간 변화",
    "tr_uo_rate_6h": "최근 6시간 소변량(mL/kg/h)", "tr_uo_rate_12h": "최근 12시간 소변량(mL/kg/h)",
    "tr_uo_rate_24h": "최근 24시간 소변량(mL/kg/h)", "tr_uo_trend_12h": "소변량 변화(최근 12시간-그 전 12시간)",
    "tr_uo_low_h_12h": "최근 12시간 소변 저하(<0.5) 시간 수", "tr_uo_low_run_end": "현재까지 연속 소변 저하 시간",
    "tr_uo_obs_12h": "최근 12시간 소변량 기록 시간 수",
    "tr_mbp_min_12h": "최근 12시간 평균동맥압 최저", "tr_mbp_lt65_h_24h": "최근 24시간 평균동맥압 65 미만 시간 수",
    "tr_intake_24h": "최근 24시간 수액·약물 섭취량", "tr_urine_ml_24h": "최근 24시간 소변량(mL)",
    "tr_fluid_bal_24h": "최근 24시간 수분 균형(mL)", "tr_vaso_h_12h": "최근 12시간 승압제 시간",
}
UNIT_BASE = {"mbp": "mmHg", "sbp": "mmHg", "dbp": "mmHg", "hr": "회/분", "rr": "회/분", "temp_c": "°C", "bun": "mg/dL",
             "creatinine": "mg/dL", "k": "mEq/L", "na": "mEq/L", "cl": "mEq/L", "hco3": "mEq/L", "mg": "mg/dL",
             "phosphate": "mg/dL", "urine_creat": "mg/dL", "weight_daily": "kg"}


def group_of(n):
    if n.startswith("tr_"):
        return "추세(tr_)"
    if n.startswith("atc3_"):
        return "약 계열 ATC3 플래그"
    if n.startswith("atc_") or n == "drug_pip_tazo" or n == "n_kidney_relevant_classes":
        return "신독성 약 10군"
    if n.startswith("egfr_") or n in ("creat_baseline_7d", "creat_last_pre_icu") or any(n.startswith(p + "_") for p in LABS):
        return "검사(크레아티닌·eGFR·전해질 등)"
    if any(n.startswith(p + "_") for p in VITALS):
        return "활력징후"
    if n.startswith(UF) or n.startswith("weight_daily") or n == "urine_ml_kg_h_overall":
        return "소변량·수액·체중"
    return "정적·치료력(나이, 병동, 투석, 승압제 등)"


def unit_of(n, kind):
    if kind == "범주":
        return "범주"
    if kind == "0/1":
        return "0/1"
    if n.startswith("atc"):
        return "0/1"
    if n.endswith("_n_hours_obs"):
        return "시간(기록된 시간 수)"
    if n.startswith("tr_"):
        if n.endswith("_h_12h") or n.endswith("_h_24h") or n in ("tr_uo_low_run_end", "tr_cr_hsl", "tr_uo_obs_12h"):
            return "시간"
        if "uo_rate" in n or n == "tr_uo_trend_12h":
            return "mL/kg/h"
        if "ratio" in n:
            return "비(무차원)"
        if n in ("tr_cr_n",):
            return "회"
        if n.startswith(("tr_cr", "tr_bun_delta")):
            return "mg/dL" if n != "tr_cr_slope_day" else "mg/dL/일"
        if n.startswith(("tr_k", "tr_hco3")):
            return "mEq/L"
        if n == "tr_mbp_min_12h":
            return "mmHg"
        if n in ("tr_intake_24h", "tr_urine_ml_24h", "tr_fluid_bal_24h"):
            return "mL"
        return ""
    if n.startswith("egfr_"):
        return "mL/min/1.73m²"
    if n.startswith("urine_ml_kg_h"):
        return "mL/kg/h"
    if n in ("age_at_icu",):
        return "세"
    if n in ("weight_kg",):
        return "kg"
    if n.endswith(("_ml_total", "_ml")) or n in ("fluid_balance_ml",):
        return "mL"
    if n in ("vasopressor_hours", "dialysis_hours", "crrt_hours", "first_dialysis_hour"):
        return "시간"
    if n in ("creat_baseline_7d", "creat_last_pre_icu"):
        return "mg/dL"
    if n == "n_dialysis_sessions_before_t0":
        return "회"
    if n == "n_kidney_relevant_classes":
        return "개(0~10)"
    m = re.match(r"([a-z0-9_]+?)_(mean|min|max|last)$", n)
    if m and m.group(1) in UNIT_BASE:
        return UNIT_BASE[m.group(1)]
    return ""


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    ax = abs(x)
    return "%.0f" % x if ax >= 100 else ("%.1f" % x if ax >= 10 else "%.2f" % x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(HERE, "_work", "team_fixed"))
    ap.add_argument("--split", default="split_random")
    ap.add_argument("--config", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"))
    ap.add_argument("--dict", default=os.path.join(HERE, "team_pipeline", "docs", "feature_dictionary_kr.md"))
    ap.add_argument("--contrib", default=os.path.join(HERE, "feature_contrib", "feature_contrib_all.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "feature_contrib", "feature_catalog.csv"))
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    feats, cats = cfg["features"], cfg["categorical"]
    label_col = json.load(open(os.path.join(a.team_dir, "default_features.json"), encoding="utf-8"))["label"]
    wide = pd.read_parquet(os.path.join(a.team_dir, "wide.parquet"))
    tf = pd.read_parquet(os.path.join(a.team_dir, "trend_features.parquet"))
    df = wide.merge(tf, on="stay_id", how="left")
    y = df[label_col].astype(int).values

    desc = {}
    for l in open(a.dict, encoding="utf-8"):
        m = re.match(r"\|\s*`([^`]+)`\s*\|", l)
        if m:
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) >= 3:
                txt = re.sub(r"\s*출처:.*$", "", c[2].replace("`", "")).strip()
                desc[m.group(1)] = txt
    desc.update(TREND_KR)
    sh = pd.read_csv(a.contrib, encoding="utf-8-sig").set_index("feature")

    rows = []
    for i, n in enumerate(feats, 1):
        v = df[n]
        r = {"번호": i, "변수명": n, "계열": group_of(n), "설명": desc.get(n, "")}
        if n in cats:
            kind = "범주"
        else:
            vn = pd.to_numeric(v.astype("float64") if pd.api.types.is_bool_dtype(v) else v, errors="coerce")
            u = vn.dropna().unique()
            if len(u) <= 2 and set(np.round(u, 6)) <= {0.0, 1.0}:
                kind = "0/1"
            elif len(u) <= 60 and np.all(np.mod(u, 1) == 0):
                kind = "정수"
            else:
                kind = "수치"
        r["형식"] = kind
        r["단위"] = unit_of(n, kind)
        r["결측(%)"] = round(100 * v.isna().mean(), 1)
        if kind == "범주":
            vc = v.astype(object).where(v.notna()).value_counts()
            r["범주별 값과 비율"] = " / ".join("%s %.1f%%" % (k, 100 * c / len(df)) if c >= MIN_CELL else "%s <10건" % k for k, c in vc.items())
            r["고유값 수"] = int(vc.shape[0])
        else:
            x = vn.dropna()
            r["고유값 수"] = int(x.nunique())
            if kind == "0/1":
                r["1 의 비율(%)"] = round(100 * float(x.mean()), 1)
                r["AKI 비율: 0 / 1 (%)"] = " / ".join("%.1f" % (100 * y[(vn == k).values].mean()) if (vn == k).sum() >= MIN_CELL else "<10건" for k in (0, 1))
            if len(x):
                q = np.nanpercentile(x, [1, 5, 25, 50, 75, 95, 99])
                for lab, val in zip(("p1", "p5", "p25", "중앙값", "p75", "p95", "p99"), q):
                    r[lab] = float(val)
                r["평균"] = float(x.mean())
        if n in sh.index:
            s = sh.loc[n]
            r["SHAP 순위"] = int(s["rank"])
            r["평균 |SHAP|"] = float(s["mean_abs_shap"])
            r["SHAP 비중(%)"] = float(s["shap_share_pct"])
            r["분할 횟수"] = int(s["n_splits"])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.round(5).to_csv(a.out, index=False, encoding="utf-8-sig")
    print("저장: %s (%d개 변수)" % (a.out, len(out)))
    print(out["형식"].value_counts().to_string())
    print("설명이 비어 있는 변수:", int((out["설명"] == "").sum()))


if __name__ == "__main__":
    main()
