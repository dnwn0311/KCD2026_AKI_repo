# -*- coding: utf-8 -*-
"""AKI 위험모델(runs/aki_risk_team_trend)의 보정(calibration)과 Brier 점수. 집계만 저장.
보정 전(모델 원 확률) / isotonic(검증 세트에서 학습, 시험 세트에 적용; 현재 위험점수 방식) / Platt(로지스틱) 비교.
출력: calib_out/calib_metrics.csv, calib_bins_raw.csv, calib_bins_isotonic.csv, calib_bins_platt.csv, calib_curve_points.csv
"""
import json, os
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "_work", "team_fixed"); out = os.path.join(HERE, "calib_out"); os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"), encoding="utf-8"))
feats, cats = cfg["features"], cfg["categorical"]
label_col = json.load(open(os.path.join(T, "default_features.json"), encoding="utf-8"))["label"]
wide = pd.read_parquet(os.path.join(T, "wide.parquet")); tf = pd.read_parquet(os.path.join(T, "trend_features.parquet"))
spl = pd.read_parquet(os.path.join(T, "splits.parquet"), columns=["stay_id", "split_random"])
rs = pd.read_parquet(os.path.join(T, "risk_scores_trend.parquet"), columns=["stay_id", "prob"]).rename(columns={"prob": "prob_saved"})
df = wide.merge(spl, on="stay_id").merge(tf, on="stay_id", how="left").merge(rs, on="stay_id", how="left")
df = df.rename(columns={"split_random": "split", label_col: "label"}); df["split"] = df["split"].astype(str).replace({"validation": "val"})
for c in cats:
    vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
    df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
for c in feats:
    if c in cats: continue
    if df[c].dtype == bool: df[c] = df[c].astype("int8")
    elif df[c].dtype == object: df[c] = pd.to_numeric(df[c], errors="coerce")
b = lgb.Booster(model_file=os.path.join(HERE, "runs", "aki_risk_team_trend", "model.txt"))
va, te, tr = df[df.split == "val"], df[df.split == "test"], df[df.split == "train"]
pv, pt = b.predict(va[feats]), b.predict(te[feats]); yv, yt = va.label.astype(int).values, te.label.astype(int).values
prev_tr = tr.label.astype(int).mean()
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv)
plat = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pv).reshape(-1, 1), yv)
P = {"보정 전(원 확률)": pt, "isotonic(검증에서 학습)": iso.predict(pt), "Platt(검증에서 학습)": plat.predict_proba(lg(pt).reshape(-1, 1))[:, 1]}
print("저장된 보정 점수와 isotonic 재현 일치(최대 차이): %.4f" % np.nanmax(np.abs(P["isotonic(검증에서 학습)"] - te["prob_saved"].values)))

def bins(y, p, k=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), k, labels=False)
    d = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"), events=("y", "sum")).reset_index()
    return d
def ece(y, p, k=10):
    d = bins(y, p, k); return float((d.n * (d.mean_pred - d.observed).abs()).sum() / d.n.sum())
def slope_int(y, p):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p).reshape(-1, 1), y)
    # 절편은 기울기 1로 고정했을 때: logit(관찰) = a + logit(p)
    off = lg(p); a = 0.0
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(a + off))); g = (y - pr).sum(); h = (pr * (1 - pr)).sum(); a += g / h
    return float(m.coef_[0, 0]), float(a)
rng = np.random.RandomState(0); rows = []; base_b = brier_score_loss(yt, np.full(len(yt), prev_tr))
for name, p in P.items():
    s, a = slope_int(yt, p); bs = brier_score_loss(yt, p)
    bb = [brier_score_loss(yt[i], p[i]) for i in (rng.randint(0, len(yt), len(yt)) for _ in range(1000))]
    rows.append({"방식": name, "Brier": round(bs, 4), "Brier_lo": round(np.percentile(bb, 2.5), 4), "Brier_hi": round(np.percentile(bb, 97.5), 4),
                 "Brier 기준(상수 예측=학습 유병률)": round(base_b, 4), "Brier Skill Score": round(1 - bs / base_b, 3),
                 "Log loss": round(log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6)), 4), "AUROC": round(roc_auc_score(yt, p), 3),
                 "보정 기울기(1이 이상)": round(s, 3), "보정 절편(0이 이상)": round(a, 3), "ECE(10등분)": round(ece(yt, p), 4),
                 "평균 예측": round(float(p.mean()), 4), "실제 비율": round(float(yt.mean()), 4), "n": len(yt), "AKI": int(yt.sum())})
    bins(yt, p).round(4).to_csv(os.path.join(out, "calib_bins_%s.csv" % ("raw" if "전" in name else name.split("(")[0]) ), index=False)
pd.DataFrame(rows).to_csv(os.path.join(out, "calib_metrics.csv"), index=False, encoding="utf-8-sig")
# 곡선용: 20등분
pts = []
for name, p in P.items():
    d = bins(yt, p, 20); d["방식"] = name; pts.append(d)
pd.concat(pts).round(4).to_csv(os.path.join(out, "calib_curve_points.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).T.to_string())
