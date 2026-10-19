# -*- coding: utf-8 -*-
"""
AKI 위험 예측 모델 - 팀원 전처리 기준 (입실 48시간 한 시점, 이후 24시간 안 첫 AKI)

입력: _work/team/{wide.parquet, splits.parquet, default_features.json}  (wsl_team_pipeline.sh 가 만든 것)
  - 단위: ICU 입실 1건 = 사례 1건. 라벨 label_aki_next_24h (크레아티닌 또는 소변량 KDIGO 기준)
  - 분할: 팀원 splits.parquet 의 split_random(기본) / split_temporal / split_hybrid (환자 단위)
  - 특징: 팀원이 제안한 default_features.json 의 wide 열 (라벨·시점·분할 열은 코드에서 한 번 더 걸러냄)

모델: LightGBM(주) + 로지스틱 회귀 + 단순 규칙 2개(크레아티닌/기저치, 소변량) 비교
보정: val 에서 isotonic -> test 확률에 적용. 신호등: val 위험 상위 5% 빨강, 15% 노랑 이상
위험 점수 파일: _work/team/risk_scores.parquet (stay_id, 보정 확률, 신호등, 주요 요인 3개).
  train 은 환자 단위 5겹 교차 예측(out-of-fold), val/test 는 주 모델 -> 처방 LLM 입력에 쓴다

출력: results/eval_team_risk.json (집계만, 10 미만은 <10), runs/aki_risk_team/ (모델)
사용:  python train_risk_team.py [--split split_random] [--team-dir _work/team]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import common as C

FORBIDDEN_PREFIX = ("label_", "aki_by_", "onset_", "n_outlier_", "split_")
FORBIDDEN_EXACT = {"stay_id", "subject_id", "hadm_id", "intime", "time_zero", "anchor_year_group", "weight_src",
                   "hospital_expire_flag", "esrd_chronic_dialysis", "esrd_icd_current_adm", "ckd_icd_current_adm",
                   "dialysis_hx_item"}

VAR_KR = {
    "hr": "심박수", "sbp": "수축기혈압", "dbp": "이완기혈압", "mbp": "평균동맥압", "rr": "호흡수", "temp_c": "체온",
    "weight_daily": "일일 체중", "dial_blood_flow": "투석 혈류량", "dialysate_rate": "투석액 유속",
    "ultrafiltrate_ml": "한외여과량", "creatinine": "크레아티닌", "creatinine_wb": "크레아티닌(혈액가스)", "bun": "BUN",
    "urine_creat": "요중 크레아티닌", "na": "나트륨", "k": "칼륨", "cl": "염소", "hco3": "중탄산", "phosphate": "인",
    "mg": "마그네슘", "uric_acid": "요산", "urine_ml": "소변량", "drain_ml": "배액량", "stool_ml": "대변량",
    "other_output_ml": "기타 배설량", "intake_ml": "수액·약물 섭취량", "urine_ml_kg_h": "시간당 소변량(mL/kg/h)",
    "egfr_mdrd": "eGFR(MDRD)", "egfr_cg": "eGFR(Cockcroft-Gault)", "egfr_ckdepi": "eGFR(CKD-EPI)",
}
STAT_KR = {"mean": "평균", "min": "최저", "max": "최고", "first": "처음", "last": "최근", "total": "합계"}
STATIC_KR = {
    "age_at_icu": "나이", "gender": "성별", "first_careunit": "입실 병동", "admission_type": "입원 유형",
    "weight_kg": "체중(kg)", "esrd_known_at_t0": "ESRD 이력", "ckd_icd_prior_adm": "이전 입원 만성신장병 진단",
    "creat_baseline_7d": "크레아티닌 기저치(7일 최저)", "creat_last_pre_icu": "입실 전 마지막 크레아티닌",
    "n_dialysis_sessions_before_t0": "48시간 내 투석 횟수", "first_dialysis_hour": "첫 투석 시각(입실 후 시간)",
    "vasopressor_hours": "승압제 투여 시간", "dialysis_hours": "투석 시간", "crrt_hours": "CRRT 시간",
    "urine_ml_kg_h_overall": "48시간 평균 소변량(mL/kg/h)", "fluid_balance_ml": "수분 균형(mL)",
    "rrt_before_t0": "48시간 전 신대체요법", "n_kidney_relevant_classes": "신장 관련 약물군 수",
}
DRUG_KR = {
    "aminoglycoside": "아미노글리코사이드", "vancomycin": "반코마이신", "pip_tazo": "피페라실린-타조박탐",
    "polymyxin": "폴리믹신", "amphotericin": "암포테리신 B", "nucleoside_antiviral": "항바이러스제(뉴클레오사이드)",
    "nsaid": "NSAID", "loop_diuretic": "루프 이뇨제", "ace_arb": "ACE억제제/ARB", "calcineurin": "칼시뉴린 억제제",
}


def feature_text(name, value):
    """팀원 열 이름과 값 -> 한국어 위험 요인 문구 ('소변량 합계 410', '반코마이신 투여 중')"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "%s %s" % (STATIC_KR.get(name, name), value)
    if name.startswith(("atc_", "drug_")):
        key = name.split("_", 2)[-1] if name.startswith("atc_") else name[len("drug_"):]
        label = DRUG_KR.get(key, key)
        return "%s %s" % (label, "투여 중" if v >= 0.5 else "투여 안 함") if v == v else "%s 기록 없음" % label
    if name.startswith("atc3_"):
        label = "약물군 ATC %s" % name[5:]
        return "%s %s" % (label, "투여 중" if v >= 0.5 else "없음") if v == v else "%s 기록 없음" % label
    if name in STATIC_KR:
        label = STATIC_KR[name]
        if name in ("esrd_known_at_t0", "ckd_icd_prior_adm", "rrt_before_t0"):
            return "%s %s" % (label, "있음" if v >= 0.5 else "없음")
        if v != v:
            return "%s 기록 없음" % label
        if name in ("n_kidney_relevant_classes", "n_dialysis_sessions_before_t0"):
            return "%s %d" % (label, round(v))
        if name.endswith("hours") or name.endswith("hour"):
            return "%s %.0f시간" % (label, v)
        return "%s %s" % (label, ("%.1f" % v) if abs(v) < 100 else ("%.0f" % v))
    for suf in ("_n_hours_obs",) + tuple("_" + s for s in STAT_KR):
        if name.endswith(suf):
            stem = name[: -len(suf)]
            if stem in VAR_KR:
                if suf == "_n_hours_obs":
                    return "%s 측정 %d시간" % (VAR_KR[stem], round(v)) if v == v else "%s 측정 기록 없음" % VAR_KR[stem]
                if v != v:
                    return "%s %s 기록 없음" % (VAR_KR[stem], STAT_KR[suf[1:]])
                return "%s %s %s" % (VAR_KR[stem], STAT_KR[suf[1:]], ("%.1f" % v) if abs(v) < 100 else ("%.0f" % v))
    return "%s %s" % (name, ("%.2f" % v) if v == v else "기록 없음")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return {"n": C.suppress(len(y)), "auroc": None, "auprc": None}
    return {"n": C.suppress(len(y)), "positives": C.suppress(int(y.sum())),
            "auroc": round(float(roc_auc_score(y, p)), 4), "auprc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4) if np.all((p >= 0) & (p <= 1)) else None}


def calibration(y, p, bins=10):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    out = []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum():
            out.append({"bin": b + 1, "n": C.suppress(int(m.sum())), "mean_pred": round(float(p[m].mean()), 4),
                        "observed": round(float(y[m].mean()), 4) if m.sum() >= 10 else None})
    return out


def at_threshold(y, p, thr):
    alert = p >= thr
    tp = int((alert & (y == 1)).sum())
    return {"threshold": round(float(thr), 4), "alert_rate": round(float(alert.mean()), 4),
            "sensitivity": round(tp / max(1, int((y == 1).sum())), 4),
            "ppv": round(tp / int(alert.sum()), 4) if alert.sum() >= 10 else None}


def explain_top(booster, X, feats, k=3, chunk=20000):
    out = []
    for i0 in range(0, len(X), chunk):
        Xi = X.iloc[i0:i0 + chunk]
        ctb = booster.predict(Xi, pred_contrib=True)[:, :-1]
        vals = Xi.values
        for row_c, row_v in zip(ctb, vals):
            order = [j for j in np.argsort(-row_c) if row_c[j] > 0]
            # 약을 안 쓴다/병력이 없다 같은 "없음" 표시는 위험 요인 문구로 쓰지 않는다
            top = [j for j in order if not (feats[j].startswith(("atc_", "atc3_", "drug_", "esrd_", "ckd_", "rrt_"))
                                            and row_v[j] == row_v[j] and row_v[j] < 0.5)][:k]
            out.append(", ".join(feature_text(feats[j], row_v[j]) for j in top))
    return out


def main():
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(C.WORK, "team"))
    ap.add_argument("--split", default="split_random", choices=["split_random", "split_temporal", "split_hybrid"])
    ap.add_argument("--out", default=os.path.join(C.HERE, "runs", "aki_risk_team"))
    ap.add_argument("--no-scores", action="store_true", help="위험 점수 파일(처방 LLM 용)을 만들지 않음")
    a = ap.parse_args()
    t0 = time.time()

    td = a.team_dir
    cfg = json.load(open(os.path.join(td, "default_features.json"), encoding="utf-8"))
    label_col = cfg["label"]
    feats = [c for c in cfg["wide"] if c not in FORBIDDEN_EXACT and not c.startswith(FORBIDDEN_PREFIX)]
    cats = [c for c in cfg["categorical"] if c in feats]
    ids = ["stay_id", "subject_id", "hadm_id"]
    extra = [c for c in ("aki_by_creat", "aki_by_uo", "esrd_chronic_dialysis") if c]
    cols = list(dict.fromkeys(ids + feats + [label_col] + extra))
    wide = pd.read_parquet(os.path.join(td, "wide.parquet"), columns=cols)
    spl = pd.read_parquet(os.path.join(td, "splits.parquet"), columns=["stay_id", a.split])
    df = wide.merge(spl, on="stay_id", how="inner").rename(columns={a.split: "split", label_col: "label"})
    df["split"] = df["split"].astype(str).replace({"validation": "val"})
    log("%s: %s건 (환자 %s명), 특징 %d개, 분할 %s" % (os.path.basename(td.rstrip('/\\')), f"{len(df):,}",
                                                 f"{df['subject_id'].nunique():,}", len(feats), a.split))
    for c in cats:
        cat_values = sorted(df.loc[df["split"] == "train", c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=cat_values)
    for c in feats:
        if c not in cats:
            if df[c].dtype == bool:
                df[c] = df[c].astype("int8")
            elif df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    sp = {k: df[df["split"] == k] for k in ("train", "val", "test")}
    for k, d in sp.items():
        log("  %s %s건, 양성 %.1f%%" % (k, f"{len(d):,}", 100 * d["label"].mean()))
        if len(d) == 0:
            raise SystemExit("[오류] %s 분할이 비어 있습니다" % k)
    Xtr, ytr = sp["train"][feats], sp["train"]["label"].astype(int).values
    Xva, yva = sp["val"][feats], sp["val"]["label"].astype(int).values
    Xte, yte = sp["test"][feats], sp["test"]["label"].astype(int).values

    params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=3000, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    p_va, p_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, yva)
    p_te_cal = iso.predict(p_te)
    log("LightGBM: 나무 %d개, test AUROC %.3f" % (model.best_iteration_ or 0, metrics(yte, p_te)["auroc"] or 0))

    # 로지스틱 회귀 (숫자 특징만, 결측은 train 중앙값)
    num = [c for c in feats if c not in cats]
    med = Xtr[num].median()
    mu, sd = Xtr[num].fillna(med).mean(), Xtr[num].fillna(med).std().replace(0, 1)
    z = lambda X: ((X[num].fillna(med) - mu) / sd).clip(-10, 10).values
    lr = LogisticRegression(max_iter=3000, C=0.05).fit(z(Xtr), ytr)
    p_lr = lr.predict_proba(z(Xte))[:, 1]

    # 단순 규칙: 크레아티닌 최근값/기저치, 낮은 소변량
    rules = {}
    if "creatinine_last" in Xte and "creat_baseline_7d" in Xte:
        base = Xte["creat_baseline_7d"].replace(0, np.nan)
        rules["creatinine_ratio_rule"] = metrics(yte, (Xte["creatinine_last"] / base).fillna(1.0).values)
    for col, name in (("urine_ml_kg_h_last", "low_urine_last_rule"), ("urine_ml_kg_h_overall", "low_urine_overall_rule")):
        if col in Xte:
            rules[name] = metrics(yte, -Xte[col].fillna(Xte[col].median()).values)

    # 신호등 기준 (val 상위 5% 빨강, 15% 노랑 이상)
    red_raw, yellow_raw = np.quantile(p_va, 0.95), np.quantile(p_va, 0.85)
    red, yellow = float(iso.predict([red_raw])[0]), float(iso.predict([yellow_raw])[0])
    lights = {"red": at_threshold(yte, p_te_cal, red), "yellow_or_red": at_threshold(yte, p_te_cal, yellow)}

    # 하위 집단: 라벨이 크레아티닌·소변량 중 어느 기준인지 / ESRD·만성 투석 표시 여부
    te = sp["test"].reset_index(drop=True)
    sub = {}
    if "aki_by_creat" in te and "aki_by_uo" in te:
        neg = (te["label"] == 0).values
        for name, pos in (("positive_by_creatinine", te["aki_by_creat"].astype(bool).values),
                          ("positive_by_urine_only", (te["aki_by_uo"].astype(bool) & ~te["aki_by_creat"].astype(bool)).values)):
            m = neg | pos
            if pos.sum() >= 10:
                sub[name] = metrics(pos[m].astype(int), p_te[m])
    if "esrd_chronic_dialysis" in te:
        e = te["esrd_chronic_dialysis"].astype(bool).values
        for name, m in (("esrd_flagged", e), ("not_esrd", ~e)):
            if m.sum() >= 30:
                sub[name] = metrics(yte[m], p_te[m])

    samp = Xte.sample(min(len(Xte), 5000), random_state=0)
    contrib = model.booster_.predict(samp, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)

    res = {"cohort": "팀원 전처리: ICU 입실 48시간 후 한 시점, 이후 24시간 안 첫 AKI (크레아티닌 또는 소변량 KDIGO), 분할 %s" % a.split,
           "n_features": len(feats),
           "split_sizes": {k: C.suppress(len(d)) for k, d in sp.items()},
           "positive_rate": {k: round(float(d["label"].mean()), 4) for k, d in sp.items()},
           "test": dict({"lightgbm": metrics(yte, p_te), "lightgbm_calibrated": metrics(yte, p_te_cal),
                         "logistic_regression": metrics(yte, p_lr)}, **rules),
           "val_lightgbm": metrics(yva, p_va), "test_subgroups_lightgbm": sub,
           "calibration_test": calibration(yte, p_te_cal),
           "traffic_light": {"rule": "val 위험 상위 5% = 빨강, 상위 15% = 노랑 이상", "red_threshold": round(red, 4),
                             "yellow_threshold": round(yellow, 4), "test": lights},
           "top_features_mean_abs_shap": [[k, round(float(v), 4)] for k, v in imp.head(25).items()],
           "best_iteration": int(model.best_iteration_ or 0)}

    if not a.no_scores:
        n_trees = max(50, int(model.best_iteration_ or 200))
        oof = np.zeros(len(Xtr))
        groups = sp["train"]["subject_id"].values
        for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(Xtr, ytr, groups)):
            m = lgb.LGBMClassifier(n_estimators=n_trees, **params)
            m.fit(Xtr.iloc[tr], ytr[tr])
            oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
            log("  교차 예측 %d/5" % (k + 1))
        parts = []
        for name, X, raw in (("train", Xtr, oof), ("val", Xva, p_va), ("test", Xte, p_te)):
            cal = iso.predict(raw)
            tier = np.where(cal >= red, "red", np.where(cal >= yellow, "yellow", "green"))
            parts.append(pd.DataFrame({"stay_id": sp[name]["stay_id"].values, "hadm_id": sp[name]["hadm_id"].values,
                                       "subject_id": sp[name]["subject_id"].values, "split": name,
                                       "prob": cal.astype("float32"), "tier": tier,
                                       "factors": explain_top(model.booster_, X, feats)}))
        scores = pd.concat(parts, ignore_index=True)
        scores.to_parquet(os.path.join(td, "risk_scores.parquet"), index=False)
        res["llm_input_scores"] = {
            "train": "환자 단위 5겹 교차 예측", "val_test": "주 모델", "train_oof_auroc": metrics(ytr, oof)["auroc"],
            "tier_share": {k: {t: round(float(v), 4) for t, v in
                               scores.loc[scores["split"] == k, "tier"].value_counts(normalize=True).items()}
                           for k in ("train", "val", "test")}}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "model.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(model.booster_.model_to_string())
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"features": feats, "categorical": cats, "red_threshold": red, "yellow_threshold": yellow,
                   "isotonic_x": iso.X_thresholds_.tolist(), "isotonic_y": iso.y_thresholds_.tolist(),
                   "design": res["cohort"], "lightgbm_version": lgb.__version__,
                   "test_auroc": res["test"]["lightgbm"]["auroc"]}, f, ensure_ascii=False)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(res, os.path.join(C.RESULTS, "eval_team_risk.json"))
    t = res["test"]
    log("test AUROC  LightGBM %.3f | 로지스틱 %.3f | %s" % (
        t["lightgbm"]["auroc"] or 0, t["logistic_regression"]["auroc"] or 0,
        " | ".join("%s %.3f" % (k, v["auroc"] or 0) for k, v in rules.items())))
    log("test AUPRC  LightGBM %.3f (양성 비율 %.3f)" % (t["lightgbm"]["auprc"] or 0, res["positive_rate"]["test"]))
    log("신호등 빨강: 알림 %.1f%%, 민감도 %.1f%%, 양성예측도 %s"
        % (100 * lights["red"]["alert_rate"], 100 * lights["red"]["sensitivity"], lights["red"]["ppv"]))
    if sub:
        log("하위 집단 AUROC: " + ", ".join("%s %.3f" % (k, v["auroc"] or 0) for k, v in sub.items()))
    log("완료. 집계: results/eval_team_risk.json, 모델: runs/aki_risk_team/" +
        ("" if a.no_scores else ", 점수: _work/team/risk_scores.parquet"))


if __name__ == "__main__":
    main()
