#!/usr/bin/env python3
"""Qian 2021 재현 모델의 특징 기여도(SHAP) 집계.

qian2021_replicate.py 의 `run` 과 같은 순서(결측 50% 초과 행 제거 -> MICE(선형회귀) -> 언더샘플링 -> 80/20)를
같은 시드(seed + 반복번호)로 다시 만들고, 시험 세트에서 모델별 SHAP 을 계산해 평균한다.

  LightGBM : booster.predict(pred_contrib=True)   (트리 SHAP, 로그오즈 단위)
  XGBoost  : booster.predict(pred_contribs=True)  (트리 SHAP, 로그오즈 단위)
  LR       : coef * (z - mean(z))                 (선형 모델의 정확한 SHAP, 표준화 입력)

출력은 특징별 요약(집계)만 저장한다. 환자 행은 저장하지 않는다.
사용: python qian_shap.py --table qian_table.parquet --out qian_shap_default [--strict --exclude-day1-aki]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qian2021_replicate import FEATURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="qian_table.parquet")
    ap.add_argument("--out", default="qian_shap")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--exclude-day1-aki", action="store_true")
    a = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb

    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.table)
    if a.exclude_day1_aki:
        df = df[df["day1_aki"] == 0]
    feats = [c for c in FEATURES if c in df.columns]
    df = df[df[feats].isna().mean(axis=1) <= 0.5].reset_index(drop=True)
    y_all = df["label"].values.astype(int)
    Xraw = df[feats].values.astype(float)
    print("분석 대상 %d건, AKI %d건, 특징 %d개" % (len(df), y_all.sum(), len(feats)), flush=True)

    def imputer():
        return IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0, sample_posterior=False)

    if not a.strict:
        X_imp = imputer().fit_transform(Xraw)
    pos_idx, neg_idx = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
    n_pos = len(pos_idx)
    names = ["LightGBM", "XGBoost", "LR"]
    F = len(feats)
    mabs = {m: [] for m in names}          # 반복별 mean|SHAP|
    msgn = {m: [] for m in names}          # 반복별 mean SHAP (부호 있는)
    mpos = {m: [] for m in names}          # AKI 환자 평균 SHAP
    mneg = {m: [] for m in names}
    hi = {m: [] for m in names}            # 값 상위 25% 평균 SHAP
    lo = {m: [] for m in names}
    rho = {m: [] for m in names}
    aucs = {m: [] for m in names}
    miss_rate = []

    for r in range(a.n_runs):
        seed = a.seed + r
        rng = np.random.RandomState(seed)
        sel = np.concatenate([pos_idx, rng.choice(neg_idx, n_pos, replace=False)])
        itr, ite = train_test_split(sel, test_size=0.2, stratify=y_all[sel], random_state=seed)
        if a.strict:
            imp = imputer().fit(Xraw[itr])
            Xtr, Xte = imp.transform(Xraw[itr]), imp.transform(Xraw[ite])
        else:
            Xtr, Xte = X_imp[itr], X_imp[ite]
        ytr, yte = y_all[itr], y_all[ite]
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        miss_rate.append(np.isnan(Xraw[ite]).mean(axis=0))

        g = lgb.LGBMClassifier(random_state=seed, verbose=-1).fit(Xtr, ytr)
        xg = xgb.XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0).fit(Xtr, ytr)
        lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)

        S = {}
        S["LightGBM"] = g.booster_.predict(Xte, pred_contrib=True)[:, :-1]
        S["XGBoost"] = xg.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
        S["LR"] = lr.coef_.ravel() * (Zte - Ztr.mean(axis=0))
        P = {"LightGBM": g.predict_proba(Xte)[:, 1], "XGBoost": xg.predict_proba(Xte)[:, 1], "LR": lr.predict_proba(Zte)[:, 1]}
        for m in names:
            s = S[m]
            aucs[m].append(roc_auc_score(yte, P[m]))
            mabs[m].append(np.abs(s).mean(axis=0))
            msgn[m].append(s.mean(axis=0))
            mpos[m].append(s[yte == 1].mean(axis=0))
            mneg[m].append(s[yte == 0].mean(axis=0))
            h, l, rr = np.zeros(F), np.zeros(F), np.zeros(F)
            for j in range(F):
                v = Xte[:, j]
                q1, q3 = np.quantile(v, [0.25, 0.75])
                h[j] = s[v >= q3, j].mean()
                l[j] = s[v <= q1, j].mean()
                rr[j] = spearmanr(v, s[:, j])[0] if np.std(s[:, j]) > 0 else np.nan
            hi[m].append(h); lo[m].append(l); rho[m].append(rr)
        print("  반복 %d/%d  AUC  " % (r + 1, a.n_runs) + "  ".join("%s %.3f" % (m, aucs[m][-1]) for m in names), flush=True)

    # 요약표
    rows = []
    for m in names:
        A = np.array(mabs[m])
        mean_abs = A.mean(axis=0)
        share = 100 * mean_abs / mean_abs.sum()
        ranks = np.array([pd.Series(-x).rank(method="min").values for x in A])
        for j, f in enumerate(feats):
            rows.append({
                "모델": m, "특징": f,
                "mean|SHAP|": mean_abs[j], "기여도(%)": share[j],
                "순위": int(pd.Series(-mean_abs).rank(method="min").iloc[j]),
                "순위_최소": int(ranks[:, j].min()), "순위_최대": int(ranks[:, j].max()),
                "기여도_표준편차(%p)": float(np.std(100 * A[:, j] / A.sum(axis=1), ddof=1)) if a.n_runs > 1 else np.nan,
                "AKI군 평균SHAP": np.mean(mpos[m], axis=0)[j], "비AKI군 평균SHAP": np.mean(mneg[m], axis=0)[j],
                "값 상위25% 평균SHAP": np.mean(hi[m], axis=0)[j], "값 하위25% 평균SHAP": np.mean(lo[m], axis=0)[j],
                "값-SHAP 스피어만": np.nanmean(np.array(rho[m])[:, j]),
                "시험세트 결측률(%)": 100 * np.mean(miss_rate, axis=0)[j],
            })
    out = pd.DataFrame(rows)
    out.round(5).to_csv(os.path.join(a.out, "qian_shap_features.csv"), index=False, encoding="utf-8-sig")
    # 변수 묶음(측정 항목) 단위 합산: min/max 를 한 항목으로
    def base(f):
        return f.rsplit("_", 1)[0] if f.endswith(("_min", "_max")) else f
    gr = out.assign(항목=out["특징"].map(base)).groupby(["모델", "항목"], as_index=False)["기여도(%)"].sum()
    gr["순위"] = gr.groupby("모델")["기여도(%)"].rank(ascending=False, method="min").astype(int)
    gr.sort_values(["모델", "순위"]).round(3).to_csv(os.path.join(a.out, "qian_shap_items.csv"), index=False, encoding="utf-8-sig")

    lines = ["조건: %s%s   n_runs=%d" % ("엄격(--strict)" if a.strict else "논문 방식", " + 1일차 AKI 제외" if a.exclude_day1_aki else "", a.n_runs),
             "시험세트 AUC 평균: " + "  ".join("%s %.3f" % (m, np.mean(aucs[m])) for m in names), ""]
    for m in names:
        d = out[out["모델"] == m].sort_values("순위")
        lines.append("[%s] 상위 15개  (mean|SHAP| 로그오즈 / 기여도%% / 순위범위 / 값 상위25%% vs 하위25%% 평균SHAP / 스피어만)" % m)
        for _, x in d.head(15).iterrows():
            lines.append("  %2d %-18s %.3f  %5.1f%%  [%2d-%2d]  hi %+.3f  lo %+.3f  rho %+.2f" % (
                x["순위"], x["특징"], x["mean|SHAP|"], x["기여도(%)"], x["순위_최소"], x["순위_최대"],
                x["값 상위25% 평균SHAP"], x["값 하위25% 평균SHAP"], x["값-SHAP 스피어만"]))
        lines.append("")
        g2 = gr[gr["모델"] == m].sort_values("순위")
        lines.append("[%s] 측정 항목별(min+max 합산) 기여도%%: " % m + ", ".join("%s %.1f" % (r["항목"], r["기여도(%)"]) for _, r in g2.iterrows()))
        lines.append("")
    open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("저장:", a.out)


if __name__ == "__main__":
    main()
