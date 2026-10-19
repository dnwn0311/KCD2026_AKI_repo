# -*- coding: utf-8 -*-
"""SHAP summary(beeswarm) plot. 환자 단위 점이 그려지므로 PC 안에서만 만들고 파일도 PC 폴더에만 둔다.
  team  : 팀 위험모델(LightGBM+추세, 280개), 시험 세트 1,658건, 상위 20개
  qian  : Qian 2021 재현 LightGBM(31개), 반복 1회차 시험 세트, 논문 방식 / 엄격 조건
사용: python shap_beeswarm.py team|qian|all [--top 20]
x축 = SHAP value(로그오즈; 오른쪽=AKI 예측 점수 증가), 색 = 특징 값(빨강 높음, 파랑 낮음, 회색 결측·범주형)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import shap, lightgbm as lgb
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "shap_beeswarm_out"); os.makedirs(OUT, exist_ok=True)
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f: fm.fontManager.addfont(f)
names = {f.name for f in fm.fontManager.ttflist}
KFONT = next((n for n in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC") if n in names), None)
if KFONT: plt.rcParams["font.family"] = KFONT
plt.rcParams["axes.unicode_minus"] = False
KR = {  # 팀 모델 상위 특징 한글 설명 (그림 라벨용)
 "first_careunit": "입실 병동", "urine_ml_n_hours_obs": "소변량 기록 시간 수", "tr_uo_low_h_12h": "최근12h 저소변량 시간",
 "urine_ml_kg_h_last": "마지막 소변량 속도", "tr_uo_rate_6h": "최근6h 소변량 속도", "mbp_last": "마지막 평균동맥압",
 "urine_ml_kg_h_overall": "전체 소변량 속도", "weight_kg": "체중", "atc_C03C_loop_diuretic": "루프이뇨제",
 "dbp_last": "마지막 이완기혈압", "n_kidney_relevant_classes": "신장 관련 약군 수", "tr_cr_rise_from_min": "크레아티닌 상승(최저 대비)",
 "urine_ml_kg_h_mean": "소변량 속도 평균", "egfr_ckdepi_min": "eGFR 최솟값", "age_at_icu": "나이", "tr_cr_rise_vs_ref": "크레아티닌 상승(기준 대비)",
 "temp_c_min": "최저 체온", "egfr_mdrd_last": "마지막 eGFR(MDRD)", "egfr_ckdepi_last": "마지막 eGFR(CKD-EPI)", "tr_uo_rate_12h": "최근12h 소변량 속도",
 "temp_c_mean": "평균 체온", "atc3_V06D": "V06D 계열 약", "hr_last": "마지막 심박수"}

def draw(vals, data, fnames, title, fn, top, note):
    ex = shap.Explanation(values=vals, data=data, feature_names=fnames)
    plt.figure(figsize=(10.5, 0.42 * top + 2.4))
    shap.plots.beeswarm(ex, max_display=top, show=False, color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_xlabel("SHAP value  (← AKI 예측 점수 낮춤 | 높임 →, 로그오즈)", fontsize=10)
    plt.title(title, fontsize=12); plt.gcf().text(0.01, 0.005, note, fontsize=8, color="#555")
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(os.path.join(OUT, fn), dpi=170); plt.close()

def sanity(vals, data, fnames, k=8):
    order = np.argsort(-np.abs(vals).mean(0))[:k]; print("   [점검] 평균|SHAP| 상위, 값-SHAP 순위상관(+면 값이 높을수록 점수 증가):")
    for j in order:
        v = data[:, j]; ok = ~np.isnan(v)
        r = spearmanr(v[ok], vals[ok, j])[0] if ok.sum() > 30 and np.nanstd(v) > 0 else float("nan")
        print("     %-26s |SHAP|=%.3f  rho=%+.2f  결측 %.0f%%" % (fnames[j], np.abs(vals[:, j]).mean(), r, 100 * (~ok).mean()))

def team(top):
    cfg = json.load(open(os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"), encoding="utf-8")); feats, cats = cfg["features"], cfg["categorical"]
    T = os.path.join(HERE, "_work", "team_fixed"); label_col = json.load(open(os.path.join(T, "default_features.json"), encoding="utf-8"))["label"]
    wide = pd.read_parquet(os.path.join(T, "wide.parquet")); tf = pd.read_parquet(os.path.join(T, "trend_features.parquet"))
    spl = pd.read_parquet(os.path.join(T, "splits.parquet"), columns=["stay_id", "split_random"])
    df = wide.merge(spl, on="stay_id").merge(tf, on="stay_id", how="left").rename(columns={"split_random": "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    for c in cats:
        vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique()); df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    for c in feats:
        if c in cats: continue
        if df[c].dtype == bool: df[c] = df[c].astype("int8")
        elif df[c].dtype == object: df[c] = pd.to_numeric(df[c], errors="coerce")
    te = df[df.split == "test"]; X = te[feats]
    b = lgb.Booster(model_file=os.path.join(HERE, "runs", "aki_risk_team_trend", "model.txt"))
    vals = b.predict(X, pred_contrib=True)[:, :-1]
    data = np.column_stack([np.full(len(X), np.nan) if c in cats else X[c].to_numpy(dtype=float) for c in feats])
    fn = [("%s (%s)" % (c, KR[c]) if (KFONT and c in KR) else c) for c in feats]
    print("팀 모델: 시험 %d건, 특징 %d개" % (len(X), len(feats))); sanity(vals, data, feats)
    draw(vals, data, fn, "팀 위험모델 SHAP summary plot (LightGBM+추세, 시험 %d건, 상위 %d)" % (len(X), top), "shap_beeswarm_팀위험모델.png", top,
         "회색 = 결측 또는 범주형(병동). SHAP은 모델 설명이며 인과가 아님. 점 하나 = 환자 1명.")

def qian(top, strict):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from qian2021_replicate import FEATURES
    df = pd.read_parquet(os.path.join(HERE, "qian_table.parquet"))
    if strict: df = df[df["day1_aki"] == 0]
    feats = [c for c in FEATURES if c in df.columns]; df = df[df[feats].isna().mean(axis=1) <= 0.5].reset_index(drop=True)
    y = df["label"].values.astype(int); Xr = df[feats].values.astype(float)
    imp = lambda: IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0, sample_posterior=False)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; seed = 0; rng = np.random.RandomState(seed)
    sel = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)]); itr, ite = train_test_split(sel, test_size=0.2, stratify=y[sel], random_state=seed)
    if strict: im = imp().fit(Xr[itr]); Xtr, Xte = im.transform(Xr[itr]), im.transform(Xr[ite])
    else: Xi = imp().fit_transform(Xr); Xtr, Xte = Xi[itr], Xi[ite]
    g = lgb.LGBMClassifier(random_state=seed, verbose=-1).fit(Xtr, y[itr]); vals = g.booster_.predict(Xte, pred_contrib=True)[:, :-1]
    data = Xte.copy(); data[np.isnan(Xr[ite])] = np.nan          # 보완한 값은 회색(원래 결측)으로 표시
    tag = "엄격 조건(1일차 AKI 제외, 보완은 학습만)" if strict else "논문 방식"
    print("Qian %s: 시험 %d건(균형 50:50), 특징 %d개" % (tag, len(Xte), len(feats))); sanity(vals, data, feats)
    draw(vals, data, feats, "Qian 2021 재현 LightGBM SHAP summary plot - %s (시험 %d건, 상위 %d)" % (tag, len(Xte), top),
         "shap_beeswarm_Qian_%s.png" % ("엄격" if strict else "논문방식"), top, "반복 1회차 시험 세트. 회색 = 원래 결측(보완 전). 균형화(50:50) 데이터라 실제 유병률과 다름. 인과 아님.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["team", "qian", "all"]); ap.add_argument("--top", type=int, default=20); a = ap.parse_args()
    if a.which in ("team", "all"): team(a.top)
    if a.which in ("qian", "all"): qian(a.top, False); qian(a.top, True)
    print("저장:", OUT, "폰트:", KFONT)
