# -*- coding: utf-8 -*-
"""AKI 위험 모델(입실 48h -> 72h) 시험 세트 성능 곡선 점 집계. 환자 단위 행은 저장하지 않는다.
출력(aki_perf_out/): roc_points.csv, pr_points.csv, calibration_bins.csv, headline.csv, tier_table.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "_work", "team_fixed")
out = os.path.join(HERE, "aki_perf_out"); os.makedirs(out, exist_ok=True)
r = pd.read_parquet(os.path.join(T, "risk_scores_trend.parquet"))
c = pd.read_parquet(os.path.join(T, "cohort_final.parquet"), columns=["stay_id", "y"])
d = r.merge(c, on="stay_id")
d["split"] = d["split"].astype(str).replace({"validation": "val"})
te = d[d["split"] == "test"]
y, p = te["y"].astype(int).values, te["prob"].values
rng = np.random.RandomState(0)
au = roc_auc_score(y, p); ap = average_precision_score(y, p)
b_au, b_ap = [], []
for _ in range(1000):
    i = rng.randint(0, len(y), len(y))
    if y[i].sum() == 0:
        continue
    b_au.append(roc_auc_score(y[i], p[i])); b_ap.append(average_precision_score(y[i], p[i]))
pd.DataFrame([{"split": "test", "n": len(y), "AKI": int(y.sum()), "prevalence_pct": round(100 * y.mean(), 1),
               "AUROC": round(au, 3), "AUROC_lo": round(np.percentile(b_au, 2.5), 3), "AUROC_hi": round(np.percentile(b_au, 97.5), 3),
               "AUPRC": round(ap, 3), "AUPRC_lo": round(np.percentile(b_ap, 2.5), 3), "AUPRC_hi": round(np.percentile(b_ap, 97.5), 3)}]
             ).to_csv(os.path.join(out, "headline.csv"), index=False, encoding="utf-8-sig")
fpr, tpr, _ = roc_curve(y, p)
k = np.unique(np.round(np.linspace(0, len(fpr) - 1, 150)).astype(int))
pd.DataFrame({"fpr": fpr[k], "tpr": tpr[k]}).round(4).to_csv(os.path.join(out, "roc_points.csv"), index=False)
pr, rc, _ = precision_recall_curve(y, p)
k = np.unique(np.round(np.linspace(0, len(pr) - 1, 150)).astype(int))
pd.DataFrame({"recall": rc[k], "precision": pr[k]}).round(4).to_csv(os.path.join(out, "pr_points.csv"), index=False)
q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
cal = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
cal.round(4).to_csv(os.path.join(out, "calibration_bins.csv"), index=False)
rows = []
for t in ("red", "yellow", "green"):
    m = (te["tier"].values == t)
    rows.append({"tier": t, "n": int(m.sum()), "share_pct": round(100 * m.mean(), 1), "AKI_n": int(y[m].sum()), "AKI_rate_pct": round(100 * y[m].mean(), 1) if m.sum() else np.nan})
pd.DataFrame(rows).to_csv(os.path.join(out, "tier_table.csv"), index=False)
print(open(os.path.join(out, "headline.csv"), encoding="utf-8-sig").read()); print(pd.DataFrame(rows)); print(cal.round(3))
