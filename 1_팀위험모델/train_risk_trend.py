# -*- coding: utf-8 -*-
"""
AKI 위험 모델에 '추세' 특징을 더했을 때 나아지는지 확인 (팀원 전처리 기준, 입실 48시간 한 시점)

1) 팀원 파이프라인의 hourly.parquet (체류당 0~47시간, time zero 이전 기록만)에서 추세 특징을 만든다
   크레아티닌: 최근 24시간 변화, 48시간 안 최저 대비 상승, 기저치 대비 비율, 기울기, 마지막 측정 후 경과 시간
   소변량: 최근 6/12/24시간 평균 속도, 최근 12시간 0.5 mL/kg/h 미만 시간 수, 끝에서부터 연속 저하 시간
   그 밖: BUN 변화와 BUN/크레아티닌, 평균동맥압 최저·65 미만 시간, 최근 24시간 수액·소변·수분 균형, 승압제 시간
   -> _work/team/trend_features.parquet
2) 같은 분할·같은 설정으로 기존 특징(기준) 모델과 기존+추세 모델을 학습해 test 에서 비교
   AUROC/AUPRC, 크레아티닌·소변량 양성 하위 집단, 신호등, 로지스틱 회귀, 차이의 부트스트랩 95% 구간

기존 파일(runs/aki_risk_team, risk_scores.parquet, eval_team_risk.json)은 건드리지 않는다.
출력: results/eval_team_risk_trend.json (집계만)
      --save 를 주면 runs/aki_risk_team_trend/ 와 _work/team/risk_scores_trend.parquet 도 만든다

사용 (WSL, ~/kcd_rx):
  python train_risk_trend.py
  python train_risk_trend.py --hourly ~/aki_pipeline/data/features/hourly.parquet --save
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import common as C
import train_risk_team as R

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
NEED = ["creatinine", "bun", "k", "hco3", "urine_ml_kg_h", "urine_ml", "intake_ml", "mbp", "vasopressor_active"]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def grid(h, var, ids, T):
    if var not in h:
        return np.full((len(ids), T), np.nan)
    p = h.pivot(index="stay_id", columns="hour", values=var).reindex(index=ids, columns=range(T))
    return p.to_numpy(dtype="float64")


def ffill_idx(A):
    T = A.shape[1]
    idx = np.where(~np.isnan(A), np.arange(T), -1)
    return np.maximum.accumulate(idx, axis=1)


def value_at(A, idx, h):
    j = idx[:, h]
    out = np.full(len(A), np.nan)
    ok = j >= 0
    out[ok] = A[np.where(ok)[0], j[ok]]
    return out


def wmean(A, a, b):
    return np.nanmean(A[:, a:b], axis=1)


def wsum_obs(A, a, b):
    s = np.nansum(A[:, a:b], axis=1)
    return np.where((~np.isnan(A[:, a:b])).any(axis=1), s, np.nan)


def build_trend(hourly_path, wide):
    import pyarrow.parquet as pq
    names = pq.read_schema(hourly_path).names
    cols = ["stay_id", "hour"] + [c for c in NEED if c in names]
    miss = [c for c in NEED if c not in names]
    if miss:
        log("[참고] hourly 에 없는 열: %s (해당 특징은 비워 둠)" % ", ".join(miss))
    h = pd.read_parquet(hourly_path, columns=cols)
    T = int(h["hour"].max()) + 1
    ids = np.sort(h["stay_id"].unique())
    log("hourly: 체류 %s건, 시간 칸 %d개 (0~%d시간)" % (f"{len(ids):,}", T, T - 1))
    G = {v: grid(h, v, ids, T) for v in NEED}
    f = pd.DataFrame({"stay_id": ids})
    h24 = max(0, T - 25)   # 끝(T-1)에서 24시간 전 이하의 마지막 값 (시점이 24시간 근처로 짧아도 인덱스가 음수가 되지 않게)

    cr = G["creatinine"]
    ci = ffill_idx(cr)
    cr_last = value_at(cr, ci, T - 1)
    cr_min = np.nanmin(cr, axis=1)
    f["tr_cr_n"] = (~np.isnan(cr)).sum(axis=1)
    f["tr_cr_hsl"] = np.where(ci[:, -1] >= 0, T - 1 - ci[:, -1], np.nan)
    f["tr_cr_delta_24h"] = cr_last - value_at(cr, ci, h24)
    f["tr_cr_rise_from_min"] = cr_last - cr_min
    f["tr_cr_ratio_min"] = cr_last / cr_min
    w = wide.set_index("stay_id").reindex(ids)
    pre = pd.to_numeric(w.get("creat_last_pre_icu"), errors="coerce").to_numpy(dtype="float64") if "creat_last_pre_icu" in w else np.full(len(ids), np.nan)
    base = pd.to_numeric(w.get("creat_baseline_7d"), errors="coerce").to_numpy(dtype="float64") if "creat_baseline_7d" in w else np.full(len(ids), np.nan)
    f["tr_cr_rise_vs_ref"] = cr_last - np.fmin(cr_min, pre)
    f["tr_cr_ratio_baseline"] = cr_last / np.where(base > 0, base, np.nan)
    M = ~np.isnan(cr)
    x = np.broadcast_to(np.arange(T, dtype="float64"), cr.shape)
    n, sx, sy = M.sum(1), (x * M).sum(1), np.nansum(cr, 1)
    sxx, sxy = (x * x * M).sum(1), np.nansum(cr * x, 1)
    den = n * sxx - sx * sx
    f["tr_cr_slope_day"] = np.where((n >= 2) & (den > 0), (n * sxy - sx * sy) / np.where(den > 0, den, 1) * 24, np.nan)
    run = np.minimum.accumulate(np.where(M, cr, np.inf), axis=1)
    prev = np.concatenate([np.full((len(ids), 1), np.inf), run[:, :-1]], axis=1)
    rise = np.where(M & np.isfinite(prev), cr - prev, np.nan)
    f["tr_cr_max_rise"] = np.nanmax(rise, axis=1)

    for v, name in (("bun", "tr_bun_delta_24h"), ("k", "tr_k_delta_24h"), ("hco3", "tr_hco3_delta_24h")):
        A = G[v]
        ii = ffill_idx(A)
        f[name] = value_at(A, ii, T - 1) - value_at(A, ii, h24)
        if v == "bun":
            f["tr_bun_cr_ratio"] = value_at(A, ii, T - 1) / np.where(cr_last > 0, cr_last, np.nan)

    uo = G["urine_ml_kg_h"]
    f["tr_uo_rate_6h"] = wmean(uo, T - 6, T)
    f["tr_uo_rate_12h"] = wmean(uo, T - 12, T)
    f["tr_uo_rate_24h"] = wmean(uo, T - 24, T)
    f["tr_uo_trend_12h"] = f["tr_uo_rate_12h"] - wmean(uo, T - 24, T - 12)
    f["tr_uo_obs_12h"] = (~np.isnan(uo[:, T - 12:])).sum(axis=1)
    f["tr_uo_low_h_12h"] = np.where(f["tr_uo_obs_12h"] > 0, (uo[:, T - 12:] < 0.5).sum(axis=1), np.nan)
    alive = np.ones(len(ids), bool)
    runlen = np.zeros(len(ids))
    for t in range(T - 1, -1, -1):
        low = uo[:, t] < 0.5          # 기록 없음(NaN)은 False -> 연속이 끊긴 것으로 본다
        runlen += alive & low
        alive &= low
    f["tr_uo_low_run_end"] = np.where((~np.isnan(uo)).any(axis=1), runlen, np.nan)

    mbp = G["mbp"]
    f["tr_mbp_min_12h"] = np.nanmin(mbp[:, T - 12:], axis=1)
    f["tr_mbp_lt65_h_24h"] = np.where((~np.isnan(mbp[:, T - 24:])).any(axis=1), (mbp[:, T - 24:] < 65).sum(axis=1), np.nan)
    f["tr_intake_24h"] = wsum_obs(G["intake_ml"], T - 24, T)
    f["tr_urine_ml_24h"] = wsum_obs(G["urine_ml"], T - 24, T)
    f["tr_fluid_bal_24h"] = f["tr_intake_24h"] - f["tr_urine_ml_24h"]
    f["tr_vaso_h_12h"] = np.nan_to_num(G["vasopressor_active"][:, T - 12:]).sum(axis=1)
    for c in f.columns[1:]:
        f[c] = f[c].astype("float32").replace([np.inf, -np.inf], np.nan)
    return f


def boot_diff(y, pa, pb, n=1000, seed=0):
    """같은 test 사례를 다시 뽑아 (b - a) 의 AUROC·AUPRC 차이 95% 구간"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.RandomState(seed)
    d_roc, d_pr = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d_roc.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
        d_pr.append(average_precision_score(y[i], pb[i]) - average_precision_score(y[i], pa[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"auroc_diff_ci95": q(d_roc), "auprc_diff_ci95": q(d_pr), "n_boot": len(d_roc)}


def top_k_alert(y, p, rate):
    k = max(1, int(round(rate * len(y))))
    alert = np.zeros(len(y), bool)
    alert[np.argsort(-p, kind="stable")[:k]] = True
    tp = int((alert & (y == 1)).sum())
    return k, tp, tp / max(1, int(y.sum())), tp / k


def alert_compare(y, pa, pb, rates=(0.05, 0.10, 0.15, 0.20), n=1000, seed=0):
    """두 모델이 test 환자 중 똑같이 상위 r% 에게 알림을 보낼 때 민감도·양성예측도 비교 (+ 차이의 95% 구간)"""
    rng = np.random.RandomState(seed)
    boots = [rng.randint(0, len(y), len(y)) for _ in range(n)]
    out = {}
    for r in rates:
        k, tpa, sa, pa_ = top_k_alert(y, pa, r)
        _, tpb, sb, pb_ = top_k_alert(y, pb, r)
        d = [top_k_alert(y[i], pb[i], r)[2] - top_k_alert(y[i], pa[i], r)[2] for i in boots if y[i].sum() > 0]
        out["%d%%" % round(100 * r)] = {
            "alerts": C.suppress(k), "base": {"caught": C.suppress(tpa), "sensitivity": round(sa, 4), "ppv": round(pa_, 4)},
            "trend": {"caught": C.suppress(tpb), "sensitivity": round(sb, 4), "ppv": round(pb_, 4)},
            "sensitivity_diff_ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    return out


def fit_eval(name, feats, cats, sp, params):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = sp["train"][feats], sp["train"]["label"].astype(int).values
    Xva, yva = sp["val"][feats], sp["val"]["label"].astype(int).values
    Xte, yte = sp["test"][feats], sp["test"]["label"].astype(int).values
    m = lgb.LGBMClassifier(n_estimators=3000, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    p_va, p_te = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, yva)
    num = [c for c in feats if c not in cats]
    med = Xtr[num].median()
    mu, sd = Xtr[num].fillna(med).mean(), Xtr[num].fillna(med).std().replace(0, 1)
    z = lambda X: ((X[num].fillna(med) - mu) / sd).clip(-10, 10).values
    p_lr = LogisticRegression(max_iter=3000, C=0.05).fit(z(Xtr), ytr).predict_proba(z(Xte))[:, 1]
    red_raw, yel_raw = np.quantile(p_va, 0.95), np.quantile(p_va, 0.85)
    red, yel = float(iso.predict([red_raw])[0]), float(iso.predict([yel_raw])[0])
    p_cal = iso.predict(p_te)
    te = sp["test"].reset_index(drop=True)
    sub = {}
    if "aki_by_creat" in te and "aki_by_uo" in te:
        neg = (te["label"] == 0).values
        bc = te["aki_by_creat"].fillna(False).astype(bool).values
        bu = te["aki_by_uo"].fillna(False).astype(bool).values
        for nm, pos in (("positive_by_creatinine", bc), ("positive_by_urine_only", bu & ~bc)):
            mm = neg | pos
            if pos.sum() >= 10:
                sub[nm] = R.metrics(pos[mm].astype(int), p_te[mm])
    res = {"n_features": len(feats), "best_iteration": int(m.best_iteration_ or 0),
           "test_lightgbm": R.metrics(yte, p_te), "test_lightgbm_calibrated": R.metrics(yte, p_cal),
           "test_logistic_regression": R.metrics(yte, p_lr), "val_lightgbm": R.metrics(yva, p_va),
           "test_subgroups": sub,
           "traffic_light_test": {"red": R.at_threshold(yte, p_cal, red), "yellow_or_red": R.at_threshold(yte, p_cal, yel)}}
    log("%-6s 특징 %3d개 | test AUROC %.3f AUPRC %.3f | 로지스틱 %.3f | 크레아티닌 양성 %s, 소변량만 양성 %s"
        % (name, len(feats), res["test_lightgbm"]["auroc"] or 0, res["test_lightgbm"]["auprc"] or 0,
           res["test_logistic_regression"]["auroc"] or 0,
           "%.3f" % sub["positive_by_creatinine"]["auroc"] if "positive_by_creatinine" in sub else "-",
           "%.3f" % sub["positive_by_urine_only"]["auroc"] if "positive_by_urine_only" in sub else "-"))
    return m, iso, red, yel, p_te, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(C.WORK, "team"))
    ap.add_argument("--hourly", default=os.path.expanduser("~/aki_pipeline/data/features/hourly.parquet"))
    ap.add_argument("--split", default="split_random", choices=["split_random", "split_temporal", "split_hybrid"])
    ap.add_argument("--rebuild", action="store_true", help="trend_features.parquet 이 있어도 다시 만든다")
    ap.add_argument("--save", action="store_true", help="추세 모델과 위험 점수 파일(risk_scores_trend.parquet)도 저장")
    a = ap.parse_args()
    t0 = time.time()
    td = a.team_dir
    R.STATIC_KR.update(TREND_KR)   # 주요 요인 문구에 한국어 이름 사용

    cfg = json.load(open(os.path.join(td, "default_features.json"), encoding="utf-8"))
    label_col = cfg["label"]
    base_feats = [c for c in cfg["wide"] if c not in R.FORBIDDEN_EXACT and not c.startswith(R.FORBIDDEN_PREFIX)]
    cats = [c for c in cfg["categorical"] if c in base_feats]
    ids = ["stay_id", "subject_id", "hadm_id"]
    extra = ["aki_by_creat", "aki_by_uo", "creat_last_pre_icu", "creat_baseline_7d"]
    wide_all = pd.read_parquet(os.path.join(td, "wide.parquet"))
    cols = [c for c in dict.fromkeys(ids + base_feats + [label_col] + extra) if c in wide_all]
    wide = wide_all[cols].copy()
    del wide_all

    tf_path = os.path.join(td, "trend_features.parquet")
    if os.path.exists(tf_path) and not a.rebuild:
        tf = pd.read_parquet(tf_path)
        log("추세 특징 불러옴: %s" % tf_path)
    else:
        if not os.path.exists(a.hourly):
            raise SystemExit("[오류] hourly.parquet 이 없습니다: %s\n  팀원 파이프라인 폴더의 data/features/hourly.parquet 경로를 --hourly 로 지정하세요." % a.hourly)
        tf = build_trend(a.hourly, wide)
        tf.to_parquet(tf_path, index=False)
        log("추세 특징 %d개 저장: %s" % (tf.shape[1] - 1, tf_path))
    trend_feats = [c for c in tf.columns if c != "stay_id"]

    spl = pd.read_parquet(os.path.join(td, "splits.parquet"), columns=["stay_id", a.split])
    df = wide.merge(spl, on="stay_id", how="inner").merge(tf, on="stay_id", how="left")
    df = df.rename(columns={a.split: "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    log("사례 %s건, 추세 특징 연결 %s건" % (f"{len(df):,}", f"{int(df['stay_id'].isin(tf['stay_id']).sum()):,}"))
    for c in cats:
        vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    for c in base_feats + trend_feats:
        if c not in cats:
            if df[c].dtype == bool:
                df[c] = df[c].astype("int8")
            elif df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    sp = {k: df[df["split"] == k] for k in ("train", "val", "test")}
    for k, d in sp.items():
        log("  %s %s건, 양성 %.1f%%" % (k, f"{len(d):,}", 100 * d["label"].mean()))

    params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, random_state=42)
    log("== 같은 분할·설정으로 두 모델 학습")
    _, _, _, _, p_base, r_base = fit_eval("기준", base_feats, cats, sp, params)
    m_tr, iso, red, yel, p_tr, r_tr = fit_eval("+추세", base_feats + trend_feats, cats, sp, params)
    yte = sp["test"]["label"].astype(int).values
    bd = boot_diff(yte, p_base, p_tr)
    log("추세 추가 효과 (test, 부트스트랩 95%%): AUROC %+.3f [%+.3f, %+.3f], AUPRC %+.3f [%+.3f, %+.3f]"
        % ((r_tr["test_lightgbm"]["auroc"] or 0) - (r_base["test_lightgbm"]["auroc"] or 0), *bd["auroc_diff_ci95"],
           (r_tr["test_lightgbm"]["auprc"] or 0) - (r_base["test_lightgbm"]["auprc"] or 0), *bd["auprc_diff_ci95"]))
    ac = alert_compare(yte, p_base, p_tr)
    log("== 같은 알림 비율 비교 (test %s건, 양성 %d건): 두 모델 모두 위험 상위 r%%에게 알림" % (f"{len(yte):,}", int(yte.sum())))
    log("  알림비율  알림수 | 기준: 잡은 수 민감도 양성예측도 | +추세: 잡은 수 민감도 양성예측도 | 민감도 차이 95% 구간")
    for r, v in ac.items():
        b, t = v["base"], v["trend"]
        log("  %5s  %6s | %5s %5.1f%% %5.1f%% | %5s %5.1f%% %5.1f%% | [%+.1f%%p, %+.1f%%p]"
            % (r, v["alerts"], b["caught"], 100 * b["sensitivity"], 100 * b["ppv"], t["caught"], 100 * t["sensitivity"],
               100 * t["ppv"], 100 * v["sensitivity_diff_ci95"][0], 100 * v["sensitivity_diff_ci95"][1]))

    feats = base_feats + trend_feats
    Xs = sp["test"][feats].sample(min(len(sp["test"]), 5000), random_state=0)
    contrib = m_tr.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
    top = [[k, round(float(v), 4), k in trend_feats] for k, v in imp.head(20).items()]
    log("+추세 모델 중요 특징 상위 10: " + ", ".join(("*" if t else "") + R.STATIC_KR.get(k, k) for k, _, t in top[:10]) + "   (* = 새 추세 특징)")

    res = {"design": "팀원 전처리(입실 48시간 한 시점, 이후 24시간 첫 AKI) + hourly 에서 만든 추세 특징, 분할 %s" % a.split,
           "trend_features": trend_feats, "base": r_base, "trend": r_tr, "trend_minus_base_bootstrap": bd, "same_alert_rate_test": ac,
           "top_features_trend_model": top}

    if a.save:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
        out = os.path.join(C.HERE, "runs", "aki_risk_team_trend")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "model.txt"), "w", encoding="utf-8", newline="\n") as fo:
            fo.write(m_tr.booster_.model_to_string())
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fo:
            json.dump({"features": feats, "categorical": cats, "red_threshold": red, "yellow_threshold": yel,
                       "isotonic_x": iso.X_thresholds_.tolist(), "isotonic_y": iso.y_thresholds_.tolist(),
                       "design": res["design"], "lightgbm_version": lgb.__version__,
                       "test_auroc": r_tr["test_lightgbm"]["auroc"]}, fo, ensure_ascii=False)
        Xtr, ytr = sp["train"][feats], sp["train"]["label"].astype(int).values
        oof = np.zeros(len(Xtr))
        n_trees = max(50, int(m_tr.best_iteration_ or 200))
        for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=5).split(Xtr, ytr, sp["train"]["subject_id"].values)):
            mk = lgb.LGBMClassifier(n_estimators=n_trees, **params).fit(Xtr.iloc[i_tr], ytr[i_tr])
            oof[i_va] = mk.predict_proba(Xtr.iloc[i_va])[:, 1]
            log("  교차 예측 %d/5" % (k + 1))
        parts = []
        for name, raw in (("train", oof), ("val", m_tr.predict_proba(sp["val"][feats])[:, 1]), ("test", p_tr)):
            cal = iso.predict(raw)
            parts.append(pd.DataFrame({"stay_id": sp[name]["stay_id"].values, "hadm_id": sp[name]["hadm_id"].values,
                                       "subject_id": sp[name]["subject_id"].values, "split": name,
                                       "prob": cal.astype("float32"),
                                       "tier": np.where(cal >= red, "red", np.where(cal >= yel, "yellow", "green")),
                                       "factors": R.explain_top(m_tr.booster_, sp[name][feats], feats)}))
        pd.concat(parts, ignore_index=True).to_parquet(os.path.join(td, "risk_scores_trend.parquet"), index=False)
        res["train_oof_auroc"] = R.metrics(ytr, oof)["auroc"]
        log("저장: runs/aki_risk_team_trend/, _work/team/risk_scores_trend.parquet (기존 risk_scores.parquet 는 그대로)")

    res["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(res, os.path.join(C.RESULTS, "eval_team_risk_trend.json"))
    log("완료 (%.1f분). 집계: results/eval_team_risk_trend.json" % res["minutes"])


if __name__ == "__main__":
    main()
