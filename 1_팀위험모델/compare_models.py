# -*- coding: utf-8 -*-
"""
위험 모델 알고리즘 비교: LightGBM vs XGBoost vs 랜덤포레스트 vs 로지스틱 회귀 (같은 자료·특징·분할)

- 특징: 팀원 전처리 요약 특징(시간별 값의 최솟값·최댓값·평균·마지막 값·측정 횟수 등) + 정적 변수 + 약물 플래그 + 추세 특징 26개
  (train_risk_trend.py 의 '+추세' 모델과 같은 특징 집합)
- 분할: 환자 단위 split_random (train/val/test), 같은 test 사례로 비교
- 학습: 부스팅 두 모델은 val AUROC 로 조기 종료, 시드 3개로 반복해 흔들림을 같이 보고
- 비교: test AUROC/AUPRC (시드 평균·범위), 같은 알림 비율(상위 5/10/15/20%)에서 잡은 AKI 환자 수,
        LightGBM 대비 AUROC 차이의 부트스트랩 95% 구간 (시드 42 기준)

기본 팀 폴더는 _work/team_fixed (NSAID·ACEi/ARB 수정본)가 있으면 그것, 없으면 _work/team.
출력: results/eval_team_risk_models.json (집계만)

사용 (WSL, kcd_venv; xgboost 가 없으면 pip install xgboost):
  cd "/mnt/c/Users/User/Documents/KCD_처방추천"
  nice -n 19 python compare_models.py
  python compare_models.py --smoke      (환자 25%만으로 점검)
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
import train_risk_trend as T

warnings.filterwarnings("ignore")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load(team_dir, split, smoke):
    """train_risk_trend.py main() 과 같은 방식으로 자료를 모은다"""
    cfg = json.load(open(os.path.join(team_dir, "default_features.json"), encoding="utf-8"))
    label_col = cfg["label"]
    base_feats = [c for c in cfg["wide"] if c not in R.FORBIDDEN_EXACT and not c.startswith(R.FORBIDDEN_PREFIX)]
    cats = [c for c in cfg["categorical"] if c in base_feats]
    ids = ["stay_id", "subject_id", "hadm_id"]
    extra = ["aki_by_creat", "aki_by_uo", "creat_last_pre_icu", "creat_baseline_7d"]
    wide_all = pd.read_parquet(os.path.join(team_dir, "wide.parquet"))
    wide = wide_all[[c for c in dict.fromkeys(ids + base_feats + [label_col] + extra) if c in wide_all]].copy()
    tf = pd.read_parquet(os.path.join(team_dir, "trend_features.parquet"))
    trend_feats = [c for c in tf.columns if c != "stay_id"]
    spl = pd.read_parquet(os.path.join(team_dir, "splits.parquet"), columns=["stay_id", split])
    df = wide.merge(spl, on="stay_id", how="inner").merge(tf, on="stay_id", how="left")
    df = df.rename(columns={split: "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    if smoke:
        keep = set(df["subject_id"].drop_duplicates().sample(frac=0.25, random_state=0))
        df = df[df["subject_id"].isin(keep)].copy()
        log("[점검 모드] 환자 25%만 사용: %d행" % len(df))
    for c in cats:
        vals = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    feats = base_feats + trend_feats
    for c in feats:
        if c not in cats:
            if df[c].dtype == bool:
                df[c] = df[c].astype("int8")
            elif df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    sp = {k: df[df["split"] == k] for k in ("train", "val", "test")}
    return sp, feats, cats, trend_feats


def to_numeric_matrix(sp, feats, cats):
    """랜덤포레스트·로지스틱용: 범주는 코드로, 결측은 train 중앙값으로"""
    out = {}
    num = [c for c in feats if c not in cats]
    med = sp["train"][num].median()
    for k, d in sp.items():
        X = d[feats].copy()
        for c in cats:
            X[c] = X[c].cat.codes.astype("float64")
        X[num] = X[num].astype("float64").fillna(med)
        out[k] = X.fillna(-1).values
    return out


def run_models(sp, feats, cats, seeds, threads, skip=()):
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    ytr, yva, yte = (sp[k]["label"].astype(int).values for k in ("train", "val", "test"))
    preds = {}   # name -> {seed: p_test}
    t0 = time.time()

    # LightGBM (train_risk_trend.py 와 같은 설정)
    preds["LightGBM"] = {}
    for s in seeds:
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, random_state=s, n_jobs=threads)
        m.fit(sp["train"][feats], ytr, eval_set=[(sp["val"][feats], yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        preds["LightGBM"][s] = m.predict_proba(sp["test"][feats])[:, 1]
    log("LightGBM 끝 (%.1f분)" % ((time.time() - t0) / 60))

    # XGBoost (비슷한 크기의 트리: 깊이 4, 같은 학습률·표본·특징 비율)
    if "XGBoost" not in skip:
        try:
            import xgboost as xgb
            preds["XGBoost"] = {}
            for s in seeds:
                m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=3, subsample=0.8,
                                      colsample_bytree=0.5, reg_lambda=2.0, tree_method="hist", enable_categorical=True,
                                      eval_metric="auc", early_stopping_rounds=150, random_state=s, n_jobs=threads, verbosity=0)
                m.fit(sp["train"][feats], ytr, eval_set=[(sp["val"][feats], yva)], verbose=False)
                preds["XGBoost"][s] = m.predict_proba(sp["test"][feats])[:, 1]
            log("XGBoost 끝 (%.1f분)" % ((time.time() - t0) / 60))
        except ImportError:
            log("[건너뜀] xgboost 가 없습니다: pip install xgboost")

    X = to_numeric_matrix(sp, feats, cats)
    preds["RandomForest"] = {}
    for s in seeds:
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=threads, random_state=s)
        m.fit(X["train"], ytr)
        preds["RandomForest"][s] = m.predict_proba(X["test"])[:, 1]
    log("랜덤포레스트 끝 (%.1f분)" % ((time.time() - t0) / 60))

    # 로지스틱 회귀 (범주 변수 제외, 표준화) - train_risk_trend.py 와 같은 방식
    num = [c for c in feats if c not in cats]
    med = sp["train"][num].median()
    Z = {k: sp[k][num].astype("float64").fillna(med) for k in sp}
    mu, sd = Z["train"].mean(), Z["train"].std().replace(0, 1)
    z = lambda k: ((Z[k] - mu) / sd).clip(-10, 10).values
    preds["LogisticRegression"] = {seeds[0]: LogisticRegression(max_iter=3000, C=0.05).fit(z("train"), ytr).predict_proba(z("test"))[:, 1]}
    log("로지스틱 끝 (%.1f분)" % ((time.time() - t0) / 60))
    return preds, yte


def summarize(preds, yte, seeds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ref = preds["LightGBM"][seeds[0]]
    res = {}
    for name, ps in preds.items():
        a = [float(roc_auc_score(yte, p)) for p in ps.values()]
        b = [float(average_precision_score(yte, p)) for p in ps.values()]
        p0 = ps[seeds[0]]
        caught = {}
        for r in (0.05, 0.10, 0.15, 0.20):
            k, tp, sens, ppv = T.top_k_alert(yte, p0, r)
            caught["%d%%" % round(100 * r)] = {"alerts": k, "caught": tp, "sensitivity": round(sens, 4), "ppv": round(ppv, 4)}
        res[name] = {"n_seeds": len(a), "auroc_mean": round(float(np.mean(a)), 4), "auroc_min": round(min(a), 4), "auroc_max": round(max(a), 4),
                     "auprc_mean": round(float(np.mean(b)), 4), "auroc_seed%d" % seeds[0]: round(float(roc_auc_score(yte, p0)), 4),
                     "same_alert_rate_seed%d" % seeds[0]: caught}
        if name != "LightGBM":
            res[name]["minus_lightgbm_bootstrap"] = T.boot_diff(yte, ref, p0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=None)
    ap.add_argument("--split", default="split_random", choices=["split_random", "split_temporal", "split_hybrid"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    td = a.team_dir or (os.path.join(C.WORK, "team_fixed") if os.path.exists(os.path.join(C.WORK, "team_fixed", "wide.parquet"))
                        else os.path.join(C.WORK, "team"))
    seeds = [int(s) for s in a.seeds.split(",")]
    log("팀 폴더: %s | 시드 %s | 스레드 %d" % (td, seeds, a.threads))
    sp, feats, cats, trend_feats = load(td, a.split, a.smoke)
    for k, d in sp.items():
        log("  %s %s건, 양성 %.1f%%" % (k, f"{len(d):,}", 100 * d["label"].mean()))
    log("특징 %d개 (추세 %d개 포함)" % (len(feats), len(trend_feats)))
    preds, yte = run_models(sp, feats, cats, seeds, a.threads)
    res = summarize(preds, yte, seeds)
    log("== test %s건, 양성 %d건 | 같은 특징·같은 분할" % (f"{len(yte):,}", int(yte.sum())))
    log("  모델                AUROC 평균 [범위]          AUPRC | LightGBM 대비 AUROC 차이 95% | 상위 5%/10%/15%/20% 잡은 수")
    for name, r in res.items():
        bd = r.get("minus_lightgbm_bootstrap")
        cs = r["same_alert_rate_seed%d" % seeds[0]]
        log("  %-18s %.3f [%.3f-%.3f]  %.3f | %s | %s" % (
            name, r["auroc_mean"], r["auroc_min"], r["auroc_max"], r["auprc_mean"],
            "[%+.3f, %+.3f]" % tuple(bd["auroc_diff_ci95"]) if bd else "      (기준)       ",
            "/".join(str(cs[k]["caught"]) for k in ("5%", "10%", "15%", "20%"))))
    if not a.smoke:
        out = os.path.join(C.HERE, "results", "eval_team_risk_models.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"design": "같은 특징(요약+정적+약물+추세)·같은 분할로 알고리즘 비교", "team_dir": td, "seeds": seeds,
                   "n_test": int(len(yte)), "n_positive": int(yte.sum()), "models": res},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log("저장: %s" % out)
    else:
        log("점검 완료. 결과 파일은 만들지 않았습니다.")


if __name__ == "__main__":
    main()
