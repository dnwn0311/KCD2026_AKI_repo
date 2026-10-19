# -*- coding: utf-8 -*-
"""
위험 예측 모델의 상위 변수별 '값의 예시'를 집계로 만든다 (환자 단위 행은 저장·출력하지 않고, 10건 미만 칸은 가린다).
  - 분포 : 전체 코호트(11,000건)의 5% / 25% / 중앙값 / 75% / 95%, 결측 비율, AKI 군과 비AKI 군의 중앙값
  - 위험 기울기 : 값을 3등분(또는 값 종류별)했을 때 실제 AKI 비율
  - 모델이 본 방향 : 테스트셋에서 SHAP 이 위험을 가장 크게 올린 상위 25% 환자와 가장 크게 낮춘 하위 25% 환자의 중앙값
  - first_careunit : 병동별 건수, AKI 비율, 평균 SHAP
실행 (feature_contrib.py 를 먼저 돌린 뒤):  python3 feature_values.py   [--top 30]
결과: feature_contrib/feature_value_examples.csv, first_careunit_by_category.csv, feature_value_examples.txt
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_CELL = 10
INFO = {  # 변수: (한국어 설명, 단위)
    "first_careunit": ("첫 ICU 병동", "범주"),
    "urine_ml_n_hours_obs": ("소변량이 기록된 시간 수 (0~47시간 중)", "시간"),
    "tr_uo_low_h_12h": ("최근 12시간 중 소변량 0.5 미만(mL/kg/h)이었던 시간 수", "시간(0~12)"),
    "urine_ml_kg_h_last": ("마지막 시간의 소변량 속도", "mL/kg/h"),
    "tr_uo_rate_6h": ("최근 6시간 평균 소변량 속도", "mL/kg/h"),
    "tr_uo_rate_12h": ("최근 12시간 평균 소변량 속도", "mL/kg/h"),
    "mbp_last": ("마지막 평균동맥압", "mmHg"),
    "dbp_last": ("마지막 이완기혈압", "mmHg"),
    "urine_ml_kg_h_overall": ("0~47시간 전체 소변량 속도 (총소변량 ÷ 체중 ÷ 기록 시간)", "mL/kg/h"),
    "urine_ml_kg_h_mean": ("0~47시간 시간별 소변량 속도의 평균", "mL/kg/h"),
    "weight_kg": ("체중", "kg"),
    "age_at_icu": ("ICU 입실 시 나이", "세"),
    "atc_C03C_loop_diuretic": ("루프이뇨제 처방 시작 (time zero 이전)", "0/1"),
    "atc3_V06D": ("약 계열 V06D 처방 시작 (time zero 이전)", "0/1"),
    "n_kidney_relevant_classes": ("신장 관련 약물군 중 처방 시작된 군의 수", "개(0~10)"),
    "tr_cr_rise_from_min": ("현재 크레아티닌 − 48시간 최저 크레아티닌", "mg/dL"),
    "tr_cr_rise_vs_ref": ("현재 크레아티닌 − min(48시간 최저, 입실 전 최저)", "mg/dL"),
    "egfr_ckdepi_min": ("eGFR(CKD-EPI 2021) 0~47시간 최솟값", "mL/min/1.73m²"),
    "egfr_ckdepi_last": ("eGFR(CKD-EPI 2021) 마지막 값", "mL/min/1.73m²"),
    "egfr_mdrd_last": ("eGFR(MDRD) 마지막 값", "mL/min/1.73m²"),
    "temp_c_min": ("체온 0~47시간 최솟값", "°C"),
    "temp_c_mean": ("체온 0~47시간 평균", "°C"),
    "hr_last": ("마지막 심박수", "회/분"),
    "hr_n_hours_obs": ("심박수가 기록된 시간 수", "시간"),
    "rr_mean": ("호흡수 0~47시간 평균", "회/분"),
    "intake_ml_total": ("0~47시간 총 섭취량(수액·경구 등)", "mL"),
    "phosphate_last": ("마지막 인 수치", "mg/dL"),
    "na_last": ("마지막 나트륨", "mEq/L"),
    "tr_fluid_bal_24h": ("최근 24시간 섭취량 − 소변량", "mL"),
    "tr_uo_obs_12h": ("최근 12시간 중 소변량이 기록된 시간 수", "시간(0~12)"),
}


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    ax = abs(x)
    return "%.0f" % x if ax >= 100 else ("%.1f" % x if ax >= 10 else "%.2f" % x)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(HERE, "_work", "team_fixed"))
    ap.add_argument("--split", default="split_random")
    ap.add_argument("--model", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "model.txt"))
    ap.add_argument("--config", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"))
    ap.add_argument("--contrib", default=os.path.join(HERE, "feature_contrib", "feature_contrib_all.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "feature_contrib"))
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    feats, cats = cfg["features"], cfg["categorical"]
    label_col = json.load(open(os.path.join(a.team_dir, "default_features.json"), encoding="utf-8"))["label"]
    wide = pd.read_parquet(os.path.join(a.team_dir, "wide.parquet"))
    tf = pd.read_parquet(os.path.join(a.team_dir, "trend_features.parquet"))
    spl = pd.read_parquet(os.path.join(a.team_dir, "splits.parquet"), columns=["stay_id", a.split])
    df = wide.merge(spl, on="stay_id", how="inner").merge(tf, on="stay_id", how="left")
    df = df.rename(columns={a.split: "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    raw_cat = {c: df[c].astype(object).where(df[c].notna()) for c in cats}     # 원래 범주 이름 보존
    for c in cats:
        vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    for c in feats:
        if c in cats:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype("int8")
        elif df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    y_all = df["label"].astype(int).values
    te_mask = (df["split"] == "test").values
    X_te = df.loc[te_mask, feats]
    contrib = lgb.Booster(model_file=a.model).predict(X_te, pred_contrib=True)[:, :-1]
    print("전체 %d건 (AKI %.1f%%), 테스트 %d건" % (len(df), 100 * y_all.mean(), te_mask.sum()))

    rank = pd.read_csv(a.contrib, encoding="utf-8-sig").sort_values("rank")
    names = list(rank["feature"].head(a.top))
    rows, lines = [], ["[전체 코호트 %d건, AKI 비율 %.1f%%]" % (len(df), 100 * y_all.mean()), ""]
    for n in names:
        j = feats.index(n)
        desc, unit = INFO.get(n, (n, ""))
        v = df[n]
        sv = contrib[:, j]
        r = {"feature": n, "설명": desc, "단위": unit, "n": len(v), "결측(%)": round(100 * v.isna().mean(), 1)}
        if n in cats:
            lines.append("%s  (%s)  범주형" % (n, desc))
            continue
        vn = pd.to_numeric(v, errors="coerce")
        ok = vn.notna().values
        if ok.sum() == 0:
            continue
        q = np.nanpercentile(vn[ok], [5, 25, 50, 75, 95])
        r.update({"p5": q[0], "p25": q[1], "중앙값": q[2], "p75": q[3], "p95": q[4]})
        for lab, m in (("AKI 군 중앙값", ok & (y_all == 1)), ("비AKI 군 중앙값", ok & (y_all == 0))):
            r[lab] = float(np.median(vn[m])) if m.sum() >= MIN_CELL else np.nan
        xt = pd.to_numeric(X_te[n], errors="coerce").values
        okt = ~np.isnan(xt)
        hi, lo = sv >= np.percentile(sv, 75), sv <= np.percentile(sv, 25)
        r["위험 크게 올린 25% 중앙값"] = float(np.median(xt[hi & okt])) if (hi & okt).sum() >= MIN_CELL else np.nan
        r["위험 크게 낮춘 25% 중앙값"] = float(np.median(xt[lo & okt])) if (lo & okt).sum() >= MIN_CELL else np.nan
        # 위험 기울기: 값 종류가 적으면 값별, 많으면 3등분
        nun = vn[ok].nunique()
        grad = []
        if nun <= 13:
            for val, g in vn[ok].groupby(vn[ok]):
                k = int(g.shape[0])
                grad.append(("%s" % fmt(float(val)), k, 100 * y_all[g.index.values].mean() if k >= MIN_CELL else np.nan))
        else:
            cut = pd.qcut(vn[ok], 3, duplicates="drop")
            for iv, g in vn[ok].groupby(cut, observed=True):
                k = int(g.shape[0])
                grad.append(("%s~%s" % (fmt(float(g.min())), fmt(float(g.max()))), k, 100 * y_all[g.index.values].mean() if k >= MIN_CELL else np.nan))
        r["AKI 비율(값 구간별)"] = " | ".join("%s: %s(n=%d)" % (lab, ("%.1f%%" % p) if p == p else "<10건", k) for lab, k, p in grad)
        rows.append(r)
        lines.append("%-26s %s [%s]\n   분포 중앙값 %s (IQR %s~%s, 5~95%% %s~%s), 결측 %.1f%%\n   AKI군 %s / 비AKI군 %s | 모델이 위험을 올린 환자 %s / 낮춘 환자 %s\n   AKI 비율: %s"
                     % (n, desc, unit, fmt(q[2]), fmt(q[1]), fmt(q[3]), fmt(q[0]), fmt(q[4]), r["결측(%)"], fmt(r["AKI 군 중앙값"]),
                        fmt(r["비AKI 군 중앙값"]), fmt(r["위험 크게 올린 25% 중앙값"]), fmt(r["위험 크게 낮춘 25% 중앙값"]), r["AKI 비율(값 구간별)"]))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(a.out, "feature_value_examples.csv"), index=False, encoding="utf-8-sig")

    if "first_careunit" in feats:
        j = feats.index("first_careunit")
        cname = raw_cat["first_careunit"]
        te_c = cname[te_mask].values
        res = []
        for c in sorted(cname.dropna().unique()):
            m = (cname == c).values
            mt = te_c == c
            n = int(m.sum())
            res.append({"병동": c, "건수": n if n >= MIN_CELL else "<10", "비율(%)": round(100 * n / len(df), 1),
                        "AKI 비율(%)": round(100 * y_all[m].mean(), 1) if n >= MIN_CELL else "",
                        "테스트 건수": int(mt.sum()) if mt.sum() >= MIN_CELL else "<10",
                        "평균 SHAP(로그오즈)": round(float(contrib[mt, j].mean()), 3) if mt.sum() >= MIN_CELL else ""})
        cu = pd.DataFrame(res).sort_values("비율(%)", ascending=False)
        cu.to_csv(os.path.join(a.out, "first_careunit_by_category.csv"), index=False, encoding="utf-8-sig")
        lines += ["", "[first_careunit 병동별]", cu.to_string(index=False)]
    txt = "\n".join(lines)
    open(os.path.join(a.out, "feature_value_examples.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
