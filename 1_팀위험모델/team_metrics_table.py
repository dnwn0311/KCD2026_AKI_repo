# -*- coding: utf-8 -*-
"""팀 AKI 위험모델(원 확률, 시험 AUROC 0.775)의 슬라이드 표 지표: AUROC(95% CI), AUPRC, Acc, Bal.Acc, PPV, NPV, Sens, Spec, F1.
임계값은 시험 세트를 보지 않고 정한다: (A) 검증 세트 Youden 최대 / (B) 신호등 '주의+위험'(검증 상위 15%) / (C) 신호등 '위험'(검증 상위 5%).
신뢰구간 = 시험 1,658건 부트스트랩 1,000회(임계값 고정). 환자 단위 행은 저장·출력하지 않고 집계만 calib_out/team_metrics_table.csv 에 저장."""
import json, os
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "_work", "team_fixed"); out = os.path.join(HERE, "calib_out"); os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"), encoding="utf-8"))
feats, cats = cfg["features"], cfg["categorical"]
label_col = json.load(open(os.path.join(T, "default_features.json"), encoding="utf-8"))["label"]
wide = pd.read_parquet(os.path.join(T, "wide.parquet")); tf = pd.read_parquet(os.path.join(T, "trend_features.parquet"))
spl = pd.read_parquet(os.path.join(T, "splits.parquet"), columns=["stay_id", "split_random"])
df = wide.merge(spl, on="stay_id").merge(tf, on="stay_id", how="left")
df = df.rename(columns={"split_random": "split", label_col: "label"}); df["split"] = df["split"].astype(str).replace({"validation": "val"})
for c in cats:
    vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
    df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
for c in feats:
    if c in cats: continue
    if df[c].dtype == bool: df[c] = df[c].astype("int8")
    elif df[c].dtype == object: df[c] = pd.to_numeric(df[c], errors="coerce")
b = lgb.Booster(model_file=os.path.join(HERE, "runs", "aki_risk_team_trend", "model.txt"))
va, te = df[df.split == "val"], df[df.split == "test"]
pv, pt = b.predict(va[feats]), b.predict(te[feats]); yv, yt = va.label.astype(int).values, te.label.astype(int).values
print("시험 %d건, AKI %d건(%.1f%%), 검증 %d건, AKI %d건" % (len(yt), yt.sum(), 100 * yt.mean(), len(yv), yv.sum()))
def conf(y, p, t):
    pr = p >= t; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum(); tn = (~pr & (y == 0)).sum()
    return tp, fp, fn, tn
def met(y, p, t):
    tp, fp, fn, tn = conf(y, p, t); d = lambda a, c: a / c if c else np.nan
    sens, spec, ppv, npv = d(tp, tp + fn), d(tn, tn + fp), d(tp, tp + fp), d(tn, tn + fn)
    return {"Acc": d(tp + tn, len(y)), "Bal.Acc": (sens + spec) / 2, "PPV": ppv, "NPV": npv, "Sens": sens, "Spec": spec, "F1": d(2 * tp, 2 * tp + fp + fn)}
fpr, tpr, th = roc_curve(yv, pv); t_you = float(th[np.argmax(tpr - fpr)])
TH = {"A. Youden(검증 기준)": t_you, "B. 신호등 주의+위험(검증 상위 15%)": float(np.quantile(pv, 0.85)), "C. 신호등 위험(검증 상위 5%)": float(np.quantile(pv, 0.95))}
rng = np.random.RandomState(0); B = [rng.randint(0, len(yt), len(yt)) for _ in range(1000)]; B = [i for i in B if yt[i].min() != yt[i].max()]
au = roc_auc_score(yt, pt); ap = average_precision_score(yt, pt)
bau = [roc_auc_score(yt[i], pt[i]) for i in B]; bap = [average_precision_score(yt[i], pt[i]) for i in B]
ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
rows = [{"임계값": "(임계값 무관)", "지표": "AUROC", "값": au, "하한": ci(bau)[0], "상한": ci(bau)[1]}, {"임계값": "(임계값 무관)", "지표": "AUPRC", "값": ap, "하한": ci(bap)[0], "상한": ci(bap)[1]}]
for name, t in TH.items():
    m = met(yt, pt, t); bm = [met(yt[i], pt[i], t) for i in B]
    tp, fp, fn, tn = conf(yt, pt, t)
    rows.append({"임계값": name, "지표": "임계 확률", "값": t, "하한": np.nan, "상한": np.nan})
    rows.append({"임계값": name, "지표": "TP/FP/FN/TN", "값": np.nan, "하한": np.nan, "상한": np.nan, "혼동행렬": "%d/%d/%d/%d" % (tp, fp, fn, tn)})
    for k, v in m.items():
        s = [x[k] for x in bm]; s = [x for x in s if x == x]
        rows.append({"임계값": name, "지표": k, "값": v, "하한": np.percentile(s, 2.5), "상한": np.percentile(s, 97.5)})
res = pd.DataFrame(rows); res.to_csv(os.path.join(out, "team_metrics_table.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
