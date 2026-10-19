# -*- coding: utf-8 -*-
"""
72시간째 AKI 가 확인된 환자(입실 48시간 시점 이후 24시간 안 AKI)에서 '48h -> 72h 처방 변경(중단 / 신규 / 증량)'을 예측하는 ML 모델.

설계
  - 코호트: 팀원 코호트(cohort_final) 중 y == 1 (48~72시간 구간에 AKI 발생)이고 처방 변경 정답 파일(data_team_trend/*.jsonl)이 있는 입실
  - 정답: prep_rx_team.py 가 만든 label(신규처방·감량·증량·중단 약 목록), active(시점 t 에 쓰고 있던 AKI 관련 약)
  - 단위: (환자, 약) 쌍.  중단·증량 = 시점 t 에 그 약을 쓰고 있는 쌍,  신규 = 쓰고 있지 않은 쌍
  - 특징: 팀 위험모델과 같은 280개(환자 상태) + 약 종류 + 현재 쓰고 있는 AKI 관련 약 34개 여부
  - 모델: 동작(중단/신규/증량)마다 LightGBM 1개 (모든 약을 한 모델에서 학습, 약 종류는 범주 특징)
  - 평가: 환자(subject_id) 단위 5-겹 교차검증. 임계값은 학습 쪽 안쪽 3-겹 예측에서 F1 최대가 되도록 정해 시험 쪽에 적용.
  - 기여도: 각 겹 시험 쪽 쌍에 대한 LightGBM pred_contrib (로그오즈)

출력은 집계(건수, 지표, 평균 |SHAP|)만 저장한다. 환자 행은 저장·출력하지 않고, 10 미만 셀은 '<10'으로 가린다.

실행 (WSL):  python rx_ml.py [--out rx_ml_out] [--folds 5] [--trees 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feature_contrib import group_of  # noqa: E402

ACTS = {"중단": "중단", "신규": "신규처방", "증량": "증량"}


def sup(n):
    return "<10" if n < 10 else int(n)


def f1_at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc, tp, fp, fn


def best_thr(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.02, 0.995, 120)))
    best = (-1, 0.5)
    for t in qs:
        f = f1_at(y, p, t)[0]
        if f > best[0]:
            best = (f, t)
    return best[1]


def main():
    import lightgbm as lgb
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=os.path.join(HERE, "_work", "team_fixed"))
    ap.add_argument("--jsonl-dir", default=os.path.join(HERE, "data_team_trend"))
    ap.add_argument("--config", default=os.path.join(HERE, "runs", "aki_risk_team_trend", "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "rx_ml_out"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "_work"))
    ap.add_argument("--no-course", dest="course", action="store_false", help="약 투여 경과 특징을 쓰지 않음")
    ap.add_argument("--preset", choices=["small", "base"], default="small", help="small: 표본이 작아 더 강하게 규제한 설정(기본), base: 위험모델과 비슷한 설정")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    cfg = json.load(open(a.config, encoding="utf-8"))
    feats, cats = cfg["features"], cfg["categorical"]
    coh = pd.read_parquet(os.path.join(a.team_dir, "cohort_final.parquet"), columns=["stay_id", "hadm_id", "subject_id", "y"])
    wide = pd.read_parquet(os.path.join(a.team_dir, "wide.parquet"))
    tf = pd.read_parquet(os.path.join(a.team_dir, "trend_features.parquet"))
    df = coh.merge(wide.drop(columns=["subject_id", "hadm_id"], errors="ignore"), on="stay_id", how="inner").merge(tf, on="stay_id", how="left")
    for c in cats:
        vals = sorted(df[c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    for c in feats:
        if c in cats:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype("int8")
        elif df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 처방 변경 정답 (jsonl 에서 case_id, label, active 만 사용)
    recs = []
    for sp in ("train", "val", "test"):
        with open(os.path.join(a.jsonl_dir, sp + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs.append((int(r["case_id"].split("-")[1]), r["label"], r["active"]))
    rx = pd.DataFrame(recs, columns=["hadm_id", "label", "active"]).drop_duplicates("hadm_id")
    df = df.merge(rx, on="hadm_id", how="inner")
    n_all = len(df)
    A = df[df["y"].astype(int) == 1].reset_index(drop=True)
    print("전체 %d건 -> 72시간 AKI 코호트 %d건" % (n_all, len(A)), flush=True)

    # 정답 사전 = 정답 파일(prep_rx_team.py)에서 학습 데이터 기준 15건 이상 나온 약 (정답에 한 번이라도 나온 약)
    vocab = sorted({d for L in rx["label"] for k in ("신규처방", "감량", "증량", "중단") for d in L.get(k, [])})
    print("약 종류 %d개" % len(vocab), flush=True)
    A["active"] = A["active"].map(lambda L: [d for d in L if d in vocab])

    # ---- 코호트 집계
    cnt = [("72시간 AKI 코호트(입실)", len(A)), ("전체 코호트(입실)", n_all), ("약 종류(전체 사전)", len(vocab))]
    rows = []
    for name, k in (("중단", "중단"), ("신규", "신규처방"), ("증량", "증량"), ("감량", "감량")):
        pats = sum(1 for L in A["label"] if len(L.get(k, [])) > 0)
        ev = sum(len(L.get(k, [])) for L in A["label"])
        dr = len({d for L in A["label"] for d in L.get(k, [])})
        rows.append({"동작": name, "환자 수": sup(pats), "환자 비율(%)": round(100 * pats / len(A), 1), "건수(환자x약)": sup(ev), "약 종류 수": sup(dr)})
    anyc = sum(1 for L in A["label"] if any(len(L.get(k, [])) for k in ("중단", "신규처방", "증량")))
    rows.append({"동작": "중단·신규·증량 중 하나 이상", "환자 수": sup(anyc), "환자 비율(%)": round(100 * anyc / len(A), 1), "건수(환자x약)": "", "약 종류 수": ""})
    # 약별 건수
    drow = []
    for d in vocab:
        r = {"약": d, "시점 t 에 사용 중(환자)": sup(sum(d in L for L in A["active"]))}
        for name, k in (("중단", "중단"), ("신규", "신규처방"), ("증량", "증량"), ("감량", "감량")):
            r[name + "(환자)"] = sup(sum(d in L.get(k, []) for L in A["label"]))
        drow.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cohort_counts.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(drow).to_csv(os.path.join(a.out, "cohort_by_drug.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cnt, columns=["항목", "값"]).to_csv(os.path.join(a.out, "cohort_size.csv"), index=False, encoding="utf-8-sig")

    # ---- (환자, 약) 쌍 만들기
    base = A[feats].reset_index(drop=True)
    pid = np.arange(len(A))
    onflags = np.zeros((len(A), len(vocab)), dtype=np.int8)
    didx = {d: i for i, d in enumerate(vocab)}
    for i, L in enumerate(A["active"]):
        for d in L:
            onflags[i, didx[d]] = 1
    on_cols = ["on_" + d for d in vocab]

    # 약 투여 경과 특징 (시점 t 이전 정보만: 시작 시각, 이미 끝난 처방의 종료 시각, 용량)
    import pyarrow.dataset as ds
    rxt = ds.dataset(os.path.join(a.work_dir, "rx.parquet")).to_table(
        columns=["hadm_id", "drug", "start", "stop", "per24"], filter=ds.field("hadm_id").isin([int(h) for h in A["hadm_id"]])).to_pandas()
    rxt = rxt[rxt["drug"].isin(vocab)]
    rx_by = {k: g for k, g in rxt.groupby(["hadm_id", "drug"])}
    tz = pd.to_datetime(A["time_zero"]).values
    CRS = ["crs_hrs_on", "crs_n_active_rows", "crs_dose_per24", "crs_n_prior_courses", "crs_hrs_since_last_stop", "crs_hrs_since_first_use"]
    def course(i, d):
        g = rx_by.get((A["hadm_id"].iloc[i], d))
        if g is None:
            return [np.nan, 0, np.nan, 0, np.nan, np.nan]
        t = tz[i]
        st, sp = g["start"].values, g["stop"].values
        act = (st <= t) & (sp > t)
        done = sp <= t
        out = [np.nan, int(act.sum()), np.nan, int(done.sum()), np.nan, np.nan]
        if act.any():
            out[0] = float((t - st[act].min()) / np.timedelta64(1, "h"))
            out[2] = float(g["per24"].values[act][np.argmax(st[act])]) if g["per24"].notna().any() else np.nan
        if done.any():
            out[4] = float((t - sp[done].max()) / np.timedelta64(1, "h"))
        started = st <= t
        if started.any():
            out[5] = float((t - st[started].min()) / np.timedelta64(1, "h"))
        return out
    n_act = onflags.sum(axis=1)

    def build(action):
        key = ACTS[action]
        P, D, Y = [], [], []
        for i in range(len(A)):
            lab = set(A["label"].iloc[i].get(key, []))
            act = set(A["active"].iloc[i])
            for d in vocab:
                if (action == "신규") == (d in act):
                    continue          # 중단·증량은 사용 중인 약만, 신규는 사용 중이 아닌 약만
                P.append(i); D.append(didx[d]); Y.append(int(d in lab))
        P, D, Y = np.array(P), np.array(D), np.array(Y)
        X = base.iloc[P].reset_index(drop=True).copy()
        X["drug"] = pd.Categorical([vocab[j] for j in D], categories=vocab)
        X["n_active_aki_drugs"] = n_act[P]
        X = pd.concat([X, pd.DataFrame(onflags[P], columns=on_cols)], axis=1)
        if a.course:
            X = pd.concat([X, pd.DataFrame([course(i, vocab[j]) for i, j in zip(P, D)], columns=CRS)], axis=1)
        return X, Y, P, D

    def group2(n):
        if n == "drug":
            return "약 종류"
        if n.startswith("crs_"):
            return "약 투여 경과(시작 후 시간 등)"
        if n.startswith("on_") or n == "n_active_aki_drugs":
            return "현재 쓰는 AKI 관련 약"
        return group_of(n)

    if a.preset == "small":
        params = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=30, colsample_bytree=0.3,
                      subsample=0.8, subsample_freq=1, reg_lambda=5.0, n_estimators=a.trees, verbose=-1, n_jobs=2)
    else:
        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=20, colsample_bytree=0.5,
                      subsample=0.8, subsample_freq=1, reg_lambda=2.0, n_estimators=a.trees, verbose=-1, n_jobs=2)
    groups = A["subject_id"].values
    ev_rows, drug_rows, shap_rows, shap_drug_rows, dir_rows = [], [], [], [], []
    tot = {"tp": 0, "fp": 0, "fn": 0}
    for action in ("중단", "신규", "증량"):
        X, Y, P, D = build(action)
        names = list(X.columns)
        num_names = [n for n in names if n != "drug" and n not in cats]
        num_idx = [names.index(n) for n in num_names]
        print("[%s] 쌍 %d개, 정답 %d개 (%.1f%%)" % (action, len(Y), Y.sum(), 100 * Y.mean()), flush=True)
        oof = np.zeros(len(Y)); thr_used = np.zeros(len(Y)); drate = np.zeros(len(Y))
        contrib_abs = np.zeros(len(names)); contrib_pos = np.zeros(len(names)); n_pos_tot = 0
        contrib_by_drug = {}
        keep_x, keep_c = [], []
        gkf = GroupKFold(n_splits=a.folds)
        for fo, (tr, te) in enumerate(gkf.split(X, Y, groups[P])):
            m = lgb.LGBMClassifier(random_state=a.seed + fo, **params).fit(X.iloc[tr], Y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            rate_tr = pd.Series(Y[tr]).groupby(D[tr]).mean()
            drate[te] = pd.Series(D[te]).map(rate_tr).fillna(Y[tr].mean()).values       # 학습 쪽에서만 센 약별 비율
            inner = np.zeros(len(tr))
            for itr, ite in GroupKFold(n_splits=3).split(X.iloc[tr], Y[tr], groups[P][tr]):
                mi = lgb.LGBMClassifier(random_state=a.seed + fo, **params).fit(X.iloc[tr].iloc[itr], Y[tr][itr])
                inner[ite] = mi.predict_proba(X.iloc[tr].iloc[ite])[:, 1]
            thr = best_thr(Y[tr], inner)
            thr_used[te] = thr
            c = m.booster_.predict(X.iloc[te], pred_contrib=True)[:, :-1]
            contrib_abs += np.abs(c).sum(axis=0)
            contrib_pos += c[Y[te] == 1].sum(axis=0); n_pos_tot += int((Y[te] == 1).sum())
            for j in range(len(vocab)):
                mk = D[te] == j
                if mk.sum() >= 30:
                    contrib_by_drug.setdefault(j, []).append((np.abs(c[mk]).sum(axis=0), int(mk.sum())))
            keep_x.append(X.iloc[te][num_names].to_numpy(dtype=float)); keep_c.append(c[:, num_idx])
            print("   겹 %d/%d 완료 (%.0f초)" % (fo + 1, a.folds, time.time() - t0), flush=True)

        pred = oof >= thr_used
        tp = int((pred & (Y == 1)).sum()); fp = int((pred & (Y == 0)).sum()); fn = int((~pred & (Y == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        prev = Y.mean()
        # 단순 기준선 1: 전부 '변경함' 으로 예측
        f1_all = 2 * prev / (1 + prev)
        # 단순 기준선 2: 약별 전체 비율만 사용 (임계값은 정답을 보고 고른 낙관적 값)
        tb = best_thr(Y, drate + 1e-9 * np.random.RandomState(0).rand(len(Y)))
        f1_prior = f1_at(Y, drate + 1e-9 * np.random.RandomState(0).rand(len(Y)), tb)[0]
        auc = roc_auc_score(Y, oof); auprc = average_precision_score(Y, oof)
        auc_prior = roc_auc_score(Y, drate)
        ev_rows.append({"동작": action, "평가 단위(환자x약)": len(Y), "정답": sup(Y.sum()), "정답 비율(%)": round(100 * prev, 2),
                        "F1": round(f1, 3), "정밀도": round(pr, 3), "재현율": round(rc, 3),
                        "AUROC": round(auc, 3), "AUPRC": round(auprc, 3),
                        "기준선 F1(전부 변경)": round(f1_all, 3), "기준선 F1(약별 비율, 임계값 낙관)": round(f1_prior, 3),
                        "기준선 AUROC(약별 비율)": round(auc_prior, 3)})
        tot["tp"] += tp; tot["fp"] += fp; tot["fn"] += fn
        # 약별 지표
        for j, d in enumerate(vocab):
            mk = D == j
            n_pos = int(Y[mk].sum())
            if mk.sum() == 0:
                continue
            row = {"동작": action, "약": d, "평가 쌍 수": int(mk.sum()), "정답 수": sup(n_pos)}
            if n_pos >= 10 and (Y[mk] == 0).sum() >= 10:
                pj = pred[mk]; yj = Y[mk]
                tpj = int((pj & (yj == 1)).sum()); fpj = int((pj & (yj == 0)).sum()); fnj = int((~pj & (yj == 1)).sum())
                prj = tpj / (tpj + fpj) if tpj + fpj else 0.0; rcj = tpj / (tpj + fnj) if tpj + fnj else 0.0
                row.update({"F1": round(2 * prj * rcj / (prj + rcj), 3) if prj + rcj else 0.0, "정밀도": round(prj, 3), "재현율": round(rcj, 3),
                            "AUROC": round(roc_auc_score(yj, oof[mk]), 3), "정답 비율(%)": round(100 * yj.mean(), 1)})
            drug_rows.append(row)
        # 기여도
        tot_n = len(Y)
        mabs = contrib_abs / tot_n
        for i, n in enumerate(names):
            shap_rows.append({"동작": action, "특징": n, "계열": group2(n), "mean|SHAP|": mabs[i], "기여도(%)": 100 * mabs[i] / mabs.sum(),
                              "정답군 평균SHAP": contrib_pos[i] / max(n_pos_tot, 1)})
        for j, lst in contrib_by_drug.items():
            s = sum(x[0] for x in lst) / sum(x[1] for x in lst)
            for i in np.argsort(-s)[:8]:
                shap_drug_rows.append({"동작": action, "약": vocab[j], "순위": int(np.where(np.argsort(-s) == i)[0][0]) + 1,
                                       "특징": names[i], "mean|SHAP|": s[i], "해당 약 쌍 수": sum(x[1] for x in lst)})
        # 방향 (수치형만): 값 상위/하위 25% 의 평균 SHAP, 스피어만
        Xk = np.concatenate(keep_x); Ck = np.concatenate(keep_c)
        nm2 = num_names
        top = [n for n in pd.DataFrame(shap_rows).query("동작 == @action").sort_values("mean|SHAP|", ascending=False)["특징"].head(30) if n in nm2]
        for n in top:
            j = nm2.index(n); v = Xk[:, j]; ok = ~np.isnan(v)
            if ok.sum() < 50 or np.nanstd(v) == 0:
                continue
            q1, q3 = np.nanpercentile(v, [25, 75])
            hi, lo = ok & (v >= q3), ok & (v <= q1)
            dir_rows.append({"동작": action, "특징": n, "값 상위25% 평균SHAP": Ck[hi, j].mean() if hi.sum() else np.nan,
                             "값 하위25% 평균SHAP": Ck[lo, j].mean() if lo.sum() else np.nan,
                             "스피어만(값,SHAP)": spearmanr(v[ok], Ck[ok, j]).correlation, "결측(%)": 100 * (~ok).mean()})
        del keep_x, keep_c, Xk, Ck

    pr = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else 0
    rc = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else 0
    ev = pd.DataFrame(ev_rows)
    ev.loc[len(ev)] = {"동작": "전체(세 동작 합산, micro)", "F1": round(2 * pr * rc / (pr + rc), 3), "정밀도": round(pr, 3), "재현율": round(rc, 3)}
    ev.loc[len(ev)] = {"동작": "전체(세 동작 평균, macro)", "F1": round(ev["F1"].iloc[:3].mean(), 3), "정밀도": round(ev["정밀도"].iloc[:3].mean(), 3), "재현율": round(ev["재현율"].iloc[:3].mean(), 3)}
    ev.to_csv(os.path.join(a.out, "metrics_overall.csv"), index=False, encoding="utf-8-sig")
    dm = pd.DataFrame(drug_rows)
    dm.to_csv(os.path.join(a.out, "metrics_by_drug.csv"), index=False, encoding="utf-8-sig")
    S = pd.DataFrame(shap_rows)
    S = S.sort_values(["동작", "mean|SHAP|"], ascending=[True, False])
    S["순위"] = S.groupby("동작")["mean|SHAP|"].rank(ascending=False, method="first").astype(int)
    S.round(5).to_csv(os.path.join(a.out, "shap_features.csv"), index=False, encoding="utf-8-sig")
    G = S.groupby(["동작", "계열"], as_index=False)["기여도(%)"].sum().sort_values(["동작", "기여도(%)"], ascending=[True, False])
    G.round(2).to_csv(os.path.join(a.out, "shap_groups.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(shap_drug_rows).round(5).to_csv(os.path.join(a.out, "shap_by_drug.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(dir_rows).round(4).to_csv(os.path.join(a.out, "shap_direction.csv"), index=False, encoding="utf-8-sig")
    print("\n[전체 지표]\n" + ev.to_string(index=False))
    print("\n저장:", a.out, " (%.0f초)" % (time.time() - t0))


if __name__ == "__main__":
    main()
