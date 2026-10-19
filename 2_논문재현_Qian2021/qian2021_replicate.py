# -*- coding: utf-8 -*-
"""
Qian et al. 2021 (Int J Gen Med 14:623-632, "Prediction Models for AKI in ICU: A Comparative Study") 재현 코드.
논문 방식:  MIMIC ICU 입실 후 "첫 24시간" 변수(14개 검사·생체 변수의 최소·최대 + 나이·성별·BMI = 31개)로
            "첫 72시간 안" AKI(KDIGO: 1일차 최저 크레아티닌 대비 2~3일차 최고 크레아티닌 상승 등)를 예측한다.
            행의 50% 초과가 결측이면 제거 -> MICE(선형회귀) 결측 보완 -> 비AKI를 AKI 수만큼 무작위 언더샘플링(균형화)
            -> 80/20 분할 -> LR, SVM, RF, XGBoost, LightGBM, CNN 6개 모델 -> 이를 10번 반복해 AUC 평균 -> 일원분산분석.
논문에 없는 부분(하이퍼파라미터, CNN 구조, 소변량 기준의 구체 구현, BMI 계산)은 라이브러리 기본값과 합리적 가정으로 채웠고,
아래 'ASSUMPTION' 주석으로 표시했다.

두 단계로 실행한다 (환자 단위 자료는 로컬에만 두고, 화면과 보고서에는 집계만 낸다).
  1) python qian2021_replicate.py build --mimic-root <MIMIC-IV 폴더> --out qian_table.parquet
       <MIMIC-IV 폴더>/hosp/*.parquet, /icu/*.parquet (또는 .csv.gz) 구조. 예: team_pipeline/data/mimiciv/3.1_parquet
  2) python qian2021_replicate.py run --table qian_table.parquet --out qian_out
       --strict               : 결측 보완을 학습 데이터에서만 학습 (논문은 전체 데이터에 보완)
       --exclude-day1-aki     : 1일차 안에 이미 크레아티닌이 0.3 이상 오른 입실을 제외 (논문 설계에는 포함됨)
       --skip-cnn             : torch 가 없을 때
추가로 '점검(audit)'을 항상 함께 낸다: 1일차 AKI 포함 여부별 성능, 레이블 섞기, 크레아티닌 제거. (논문 점수가 신뢰할 만한지 보는 용도)
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 변수 정의 (MIMIC-IV itemid)
# ASSUMPTION: itemid 는 MIMIC-IV 3.x 기준이며 d_labitems / d_items 와 대조해 확인할 것.
LAB_ITEMS = {
    "bun": [51006], "inr": [51237], "glucose": [50931, 50809], "ptt": [51275],
    "hemoglobin": [51222, 50811], "platelet": [51265], "potassium": [50971, 50822],
    "wbc": [51301, 51300], "bicarbonate": [50882, 50803], "calcium": [50893],
    "pt": [51274], "creatinine": [50912],
}
SPO2_ITEM = 220277
WEIGHT_ITEMS = [226512, 224639]          # 입실 체중, 일별 체중 (kg)
HEIGHT_CM, HEIGHT_IN = 226730, 226707
URINE_ITEMS = [226559, 226560, 226561, 226567, 226563, 226564, 226565, 226627, 226631]   # 팀 파이프라인 config.yaml 과 동일
PHYS = list(LAB_ITEMS) + ["urine_output", "spo2"]      # 14개
FEATURES = ["age", "male", "bmi"] + [f"{v}_{s}" for v in PHYS for s in ("min", "max")]   # 3 + 28 = 31


CSV_TYPES = {"diagnoses_icd": {"icd_code": "VARCHAR"}, "labevents": {"valuenum": "DOUBLE"},
             "chartevents": {"valuenum": "DOUBLE"}, "outputevents": {"value": "DOUBLE"}}


OVERRIDE = {}      # 표 이름 -> 파일 경로 (--patients-file, --chartevents-file 로 지정)


def _reader(p, name):
    p = p.replace("'", "''")
    if p.endswith(".parquet"):
        return "read_parquet('%s')" % p
    # CSV 는 열 형식을 추측하다 틀리기 쉬워(ICD 코드가 숫자로 읽히는 등) 필요한 열만 형식을 지정한다
    t = CSV_TYPES.get(name)
    ty = ", types={%s}" % ", ".join("'%s': '%s'" % kv for kv in t.items()) if t else ""
    return "read_csv_auto('%s'%s)" % (p, ty)


def src(root, mod, name):
    if OVERRIDE.get(name):
        if not os.path.exists(OVERRIDE[name]):
            raise SystemExit("[오류] 지정한 파일이 없습니다: %s" % OVERRIDE[name])
        return _reader(OVERRIDE[name], name)
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = os.path.join(root, mod, name + ext)
        if os.path.exists(p):
            return _reader(p, name)
    raise SystemExit("[오류] %s/%s 를 찾지 못했습니다 (.parquet / .csv.gz / .csv). --mimic-root 가 hosp/ 와 icu/ 의 윗 폴더인지 확인하세요." % (mod, name))


# ---------------------------------------------------------------- 1단계: 코호트·특징·레이블
def build(a):
    import duckdb
    t0 = time.time()
    OVERRIDE.update({"patients": a.patients_file, "chartevents": a.chartevents_file})
    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % max(1, (os.cpu_count() or 4) - 1))
    R = a.mimic_root
    rep = []

    def log(s):
        rep.append(s)
        print(s, flush=True)

    con.execute("create temp view icustays as select * from %s" % src(R, "icu", "icustays"))
    con.execute("create temp view patients as select * from %s" % src(R, "hosp", "patients"))
    con.execute("create temp view dx as select * from %s" % src(R, "hosp", "diagnoses_icd"))
    con.execute("create temp view labevents as select * from %s" % src(R, "hosp", "labevents"))
    con.execute("create temp view chartevents as select * from %s" % src(R, "icu", "chartevents"))
    con.execute("create temp view outputevents as select * from %s" % src(R, "icu", "outputevents"))

    part = "subject_id" if a.first_stay_per_patient else "hadm_id"
    con.execute(f"""create temp table stays as
        select i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, p.gender,
               p.anchor_age + (extract(year from i.intime) - p.anchor_year) as age_raw
        from icustays i join patients p using (subject_id)
        qualify row_number() over (partition by i.{part} order by i.intime) = 1""")
    n0 = con.execute("select count(*) from icustays").fetchone()[0]
    n1 = con.execute("select count(*) from stays").fetchone()[0]
    log("ICU 입실 %s건 -> %s당 첫 입실 %s건" % (f"{n0:,}", "환자" if a.first_stay_per_patient else "입원", f"{n1:,}"))

    con.execute("delete from stays where age_raw < 18")
    n2 = con.execute("select count(*) from stays").fetchone()[0]
    log("나이 18세 이상: %s건 (89세 초과는 90 으로 처리)" % f"{n2:,}")

    # 제외: 만성콩팥병, 말기콩팥병, 신장이식 (ICD 진단은 퇴원 때 부여되는 코드임. 입원 시점 정보가 아님에 유의)
    con.execute("""create temp table excl as select distinct hadm_id from dx where
        (icd_version = 10 and (icd_code like 'N18%' or icd_code in ('Z940', 'Z992')))
        or (icd_version = 9 and (icd_code like '585%' or icd_code in ('V420', 'V451', 'V4511')))""")
    con.execute("delete from stays where hadm_id in (select hadm_id from excl)")
    n3 = con.execute("select count(*) from stays").fetchone()[0]
    log("만성콩팥병·말기콩팥병·신장이식 제외: %s건" % f"{n3:,}")

    # 검사값 (입실 후 72시간까지)
    lm = pd.DataFrame([(i, v) for v, ids in LAB_ITEMS.items() for i in ids], columns=["itemid", "var"])
    con.register("lab_map", lm)
    con.execute("""create temp table lab as
        select s.stay_id, m.var, l.valuenum v, date_diff('second', s.intime, l.charttime) / 3600.0 h
        from labevents l join stays s on l.hadm_id = s.hadm_id join lab_map m on l.itemid = m.itemid
        where l.valuenum is not null and l.charttime >= s.intime and l.charttime < s.intime + interval 72 hour""")
    con.execute("delete from lab where var = 'creatinine' and (v < 0.1 or v > 30)")
    log("검사값 %s건 추출 (%.0f초)" % (f"{con.execute('select count(*) from lab').fetchone()[0]:,}", time.time() - t0))

    # chartevents 는 가장 커서 한 번만 훑는다 (필요한 itemid 만, 입실 후 72시간까지)
    wi = ",".join(map(str, WEIGHT_ITEMS))
    con.execute(f"""create temp table ce as
        select s.stay_id, c.itemid, c.valuenum, c.charttime, date_diff('second', s.intime, c.charttime) / 3600.0 h
        from chartevents c join stays s using (stay_id)
        where c.itemid in ({SPO2_ITEM}, {wi}, {HEIGHT_CM}, {HEIGHT_IN}) and c.valuenum is not null""")
    log("chartevents 추출 완료 (%.0f초)" % (time.time() - t0))
    con.execute(f"""create temp table spo2 as select stay_id, valuenum v from ce
        where itemid = {SPO2_ITEM} and valuenum between 0 and 100 and h >= 0 and h < 24""")
    # 체중·키 -> BMI. ASSUMPTION: 입실 체중(없으면 일별 체중 첫 값), 키는 cm(없으면 inch*2.54)
    con.execute(f"""create temp table wt as select stay_id, arg_min(valuenum, charttime) weight
        from ce where itemid in ({wi}) and valuenum between 20 and 400 group by stay_id""")
    con.execute(f"""create temp table ht as select stay_id, arg_min(case when itemid = {HEIGHT_CM} then valuenum else valuenum * 2.54 end, charttime) height
        from ce where itemid in ({HEIGHT_CM}, {HEIGHT_IN}) and valuenum > 0 group by stay_id""")
    # 시간당 소변량(mL) : 시간 칸마다 합
    ui = ",".join(map(str, URINE_ITEMS))
    con.execute(f"""create temp table uo as
        select s.stay_id, cast(floor(date_diff('second', s.intime, o.charttime) / 3600.0) as int) bin, sum(o.value) ml
        from outputevents o join stays s using (stay_id)
        where o.itemid in ({ui}) and o.value >= 0 and o.charttime >= s.intime and o.charttime < s.intime + interval 72 hour
        group by 1, 2""")

    # 1일차 최소·최대
    feat = con.execute("""select stay_id, var, min(v) mn, max(v) mx from lab where h < 24 group by 1, 2""").df()
    f = feat.pivot(index="stay_id", columns="var", values=["mn", "mx"])
    f.columns = ["%s_%s" % (v, "min" if k == "mn" else "max") for k, v in f.columns]
    sp = con.execute("select stay_id, min(v) spo2_min, max(v) spo2_max from spo2 group by 1").df().set_index("stay_id")
    u1 = con.execute("select stay_id, min(ml) urine_output_min, max(ml) urine_output_max from uo where bin < 24 group by 1").df().set_index("stay_id")
    base = con.execute("""select s.stay_id, s.subject_id, s.hadm_id, case when s.age_raw > 89 then 90 else s.age_raw end age,
        case when s.gender = 'M' then 1 else 0 end male, w.weight, h.height
        from stays s left join wt w using (stay_id) left join ht h using (stay_id)""").df().set_index("stay_id")
    base["bmi"] = base["weight"] / (base["height"] / 100.0) ** 2
    base.loc[(base["bmi"] < 10) | (base["bmi"] > 100), "bmi"] = np.nan
    X = base[["subject_id", "hadm_id", "age", "male", "bmi", "weight"]].join(f).join(sp).join(u1)

    # 레이블: 1일차 최저 크레아티닌 대비 2~3일차(24~72시간)
    cr = con.execute("""select stay_id,
            min(v) filter (where h < 24) cr_min1, max(v) filter (where h < 24) cr_max1,
            count(*) filter (where h < 24) n1, max(v) filter (where h >= 24) cr_max23, count(*) filter (where h >= 24) n23,
            arg_min(v, h) filter (where h < 24) scr_first
        from lab where var = 'creatinine' group by 1""").df().set_index("stay_id")
    rise = con.execute("""select stay_id, max(v - rmin) rise1 from (
            select stay_id, v, min(v) over (partition by stay_id order by h, v rows between unbounded preceding and current row) rmin
            from lab where var = 'creatinine' and h < 24) group by 1""").df().set_index("stay_id")
    X = X.join(cr).join(rise)
    nA = len(X)
    X = X[(X["n1"] >= 1) & (X["n23"] >= 1)]
    log("1일차 및 2~3일차에 크레아티닌이 있는 입실: %s건 (전 단계 %s건)" % (f"{len(X):,}", f"{nA:,}"))
    X = X[X["scr_first"] < 4.0]
    log("첫 크레아티닌 4.0 미만: %s건" % f"{len(X):,}")

    by_rise = (X["cr_max23"] - X["cr_min1"]) >= 0.3
    by_ratio = X["cr_max23"] >= 1.5 * X["cr_min1"]
    # ASSUMPTION: 소변량 기준 = 24~72시간 사이 6시간 연속, 시간당 소변량/체중 < 0.5 mL/kg/h (모든 칸에 기록이 있어야 함)
    by_uo = pd.Series(False, index=X.index)
    if a.use_uo:
        u = con.execute("select stay_id, bin, ml from uo where bin >= 24 and bin < 72").df()
        u = u[u["stay_id"].isin(X.index)]
        grid = np.full((len(X), 48), np.nan)
        pos = pd.Series(np.arange(len(X)), index=X.index)
        grid[pos[u["stay_id"]].values, (u["bin"] - 24).values] = u["ml"].values
        wk = X["weight"].values[:, None]
        low = (grid / wk < 0.5) & ~np.isnan(grid) & ~np.isnan(wk)
        c = np.cumsum(np.pad(low.astype(int), ((0, 0), (1, 0))), axis=1)
        by_uo = pd.Series(((c[:, 6:] - c[:, :-6]) == 6).any(axis=1), index=X.index)
    X["label"] = (by_rise | by_ratio | by_uo).astype(int)
    X["day1_aki"] = (X["rise1"].fillna(0) >= 0.3).astype(int)   # 입실 첫 24시간 안에 이미 크레아티닌이 0.3 이상 오른 경우
    pos_n = int(X["label"].sum())
    log("최종 %s건, AKI %s건 (%.1f%%)  [크레아티닌 상승 %s, 1.5배 %s, 소변량 %s]"
        % (f"{len(X):,}", f"{pos_n:,}", 100 * pos_n / len(X), f"{int(by_rise.sum()):,}", f"{int(by_ratio.sum()):,}",
           f"{int(by_uo.sum()):,}" if a.use_uo else "미사용"))
    d1 = X["day1_aki"] == 1
    log("점검: 1일차 안에 이미 크레아티닌 0.3 이상 상승 %s건 (%.1f%%) | 그중 레이블 양성 %.1f%% | 나머지 양성 %.1f%%"
        % (f"{int(d1.sum()):,}", 100 * d1.mean(), 100 * X.loc[d1, "label"].mean() if d1.any() else float("nan"),
           100 * X.loc[~d1, "label"].mean()))
    out = X[["subject_id", "hadm_id", "label", "day1_aki"] + [c for c in FEATURES if c in X.columns]].reset_index()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    out = out[["stay_id", "subject_id", "hadm_id", "label", "day1_aki"] + FEATURES]
    out.to_parquet(a.out, index=False)
    with open(os.path.splitext(a.out)[0] + "_cohort_report.txt", "w", encoding="utf-8") as fo:
        fo.write("\n".join(rep) + "\n")
    log("저장: %s (%.0f초)" % (a.out, time.time() - t0))


# ---------------------------------------------------------------- 2단계: 균형화 · 6개 모델 · 10회 반복
def make_cnn(n_feat, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class Net(nn.Module):
        # ASSUMPTION: 논문에 구조가 없어 1차원 합성곱 2층 + 완전연결 구조로 구성
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                   nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
            self.f = nn.Sequential(nn.Flatten(), nn.Linear(64 * (n_feat // 2), 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            return self.f(self.c(x)).squeeze(-1)
    return Net()


def fit_cnn(Xtr, ytr, Xte, seed, epochs=30):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_cnn(Xtr.shape[1], seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev).unsqueeze(1)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 128):
            b = perm[i:i + 128].to(dev)
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32, device=dev).unsqueeze(1))).cpu().numpy()


def run(a):
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    import lightgbm as lgb
    import xgboost as xgb

    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.table)
    if a.exclude_day1_aki:
        n = len(df)
        df = df[df["day1_aki"] == 0]
        print("1일차 AKI 제외: %d -> %d건" % (n, len(df)))
    feats = [c for c in FEATURES if c in df.columns]
    miss = df[feats].isna().mean(axis=1)
    df = df[miss <= 0.5].reset_index(drop=True)       # 논문: 결측 50% 초과 행 제거
    y_all = df["label"].values.astype(int)
    d1_all = df["day1_aki"].values if "day1_aki" in df else np.zeros(len(df), int)
    Xraw = df[feats].values.astype(float)
    print("분석 대상 %d건, AKI %d건 (%.1f%%), 특징 %d개" % (len(df), y_all.sum(), 100 * y_all.mean(), len(feats)))

    def imputer():
        # 논문: MICE 'norm.predict' = 선형회귀 예측값으로 채움
        return IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0, sample_posterior=False)

    if not a.strict:
        X_imp = imputer().fit_transform(Xraw)          # 논문 방식: 전체 데이터에 보완
    pos_idx, neg_idx = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
    n_pos = len(pos_idx)
    if len(neg_idx) < n_pos:
        sys.exit("[오류] 비AKI 가 AKI 보다 적어 언더샘플링을 할 수 없습니다.")
    names = ["LR", "SVM", "RF", "XGBoost", "LightGBM"] + ([] if a.skip_cnn else ["CNN"])
    auc = {m: [] for m in names}
    perf = {m: [] for m in names}
    imp_w = {m: [] for m in names}
    audit = {"LightGBM 전체": [], "LightGBM 1일차 AKI 있음": [], "LightGBM 1일차 AKI 없음": [],
             "LightGBM 레이블 섞기": [], "LightGBM 크레아티닌 제거": [], "LR 크레아티닌 제거": []}
    crcols = [i for i, c in enumerate(feats) if c.startswith("creatinine_")]

    for r in range(a.n_runs):
        seed = a.seed + r
        rng = np.random.RandomState(seed)
        sel = np.concatenate([pos_idx, rng.choice(neg_idx, n_pos, replace=False)])      # 무작위 언더샘플링 -> 균형 데이터
        itr, ite = train_test_split(sel, test_size=0.2, stratify=y_all[sel], random_state=seed)   # 80/20
        if a.strict:
            imp = imputer().fit(Xraw[itr])
            Xtr, Xte = imp.transform(Xraw[itr]), imp.transform(Xraw[ite])
        else:
            Xtr, Xte = X_imp[itr], X_imp[ite]
        ytr, yte = y_all[itr], y_all[ite]
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        models = {
            "LR": LogisticRegression(max_iter=2000),
            "SVM": LinearSVC(max_iter=20000),
            "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
            "XGBoost": xgb.XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0),
            "LightGBM": lgb.LGBMClassifier(random_state=seed, verbose=-1),
        }
        for m in names:
            if m == "CNN":
                p = fit_cnn(Ztr, ytr, Zte, seed)
                pred = (p >= 0.5).astype(int)
            else:
                mdl = models[m]
                lin = m in ("LR", "SVM")
                mdl.fit(Ztr if lin else Xtr, ytr)
                Xe = Zte if lin else Xte
                if m == "SVM":
                    p = mdl.decision_function(Xe)
                    pred = (p > 0).astype(int)
                else:
                    p = mdl.predict_proba(Xe)[:, 1]
                    pred = (p >= 0.5).astype(int)
                w = np.abs(mdl.coef_).ravel() if lin else mdl.feature_importances_
                rk = np.argsort(-w)[:10]
                s = np.zeros(len(feats))
                s[rk] = np.arange(10, 0, -1)          # 논문: 상위 10개에 10~1점
                imp_w[m].append(s)
            auc[m].append(roc_auc_score(yte, p))
            perf[m].append([accuracy_score(yte, pred), precision_score(yte, pred, zero_division=0),
                            recall_score(yte, pred), f1_score(yte, pred), auc[m][-1]])

        # 점검(audit)
        gb = models["LightGBM"]
        pg = gb.predict_proba(Xte)[:, 1]
        audit["LightGBM 전체"].append(roc_auc_score(yte, pg))
        d1 = d1_all[ite]
        for k, msk in (("LightGBM 1일차 AKI 있음", d1 == 1), ("LightGBM 1일차 AKI 없음", d1 == 0)):
            if msk.sum() > 20 and len(set(yte[msk])) == 2:
                audit[k].append(roc_auc_score(yte[msk], pg[msk]))
        gp = lgb.LGBMClassifier(random_state=seed, verbose=-1).fit(Xtr, rng.permutation(ytr))
        audit["LightGBM 레이블 섞기"].append(roc_auc_score(yte, gp.predict_proba(Xte)[:, 1]))
        keep = [i for i in range(len(feats)) if i not in crcols]
        gn = lgb.LGBMClassifier(random_state=seed, verbose=-1).fit(Xtr[:, keep], ytr)
        audit["LightGBM 크레아티닌 제거"].append(roc_auc_score(yte, gn.predict_proba(Xte[:, keep])[:, 1]))
        ln = LogisticRegression(max_iter=2000).fit(Ztr[:, keep], ytr)
        audit["LR 크레아티닌 제거"].append(roc_auc_score(yte, ln.predict_proba(Zte[:, keep])[:, 1]))
        print("  반복 %d/%d  " % (r + 1, a.n_runs) + "  ".join("%s %.3f" % (m, auc[m][-1]) for m in names), flush=True)

    # 표 2: 10회 AUC 요약, 표 3: 분산분석, 표 4: 성능
    t2 = pd.DataFrame({"모델": names, "AUC 합": [sum(auc[m]) for m in names], "평균 AUC": [np.mean(auc[m]) for m in names],
                       "분산": [np.var(auc[m], ddof=1) if len(auc[m]) > 1 else np.nan for m in names]})
    t4 = pd.DataFrame([[m] + list(np.mean(perf[m], axis=0)) for m in names], columns=["모델", "정확도", "정밀도", "재현율", "F1", "AUC"])
    f_stat, p_val = stats.f_oneway(*[auc[m] for m in names]) if a.n_runs > 1 else (np.nan, np.nan)
    t2.round(6).to_csv(os.path.join(a.out, "table2_auc_summary.csv"), index=False, encoding="utf-8-sig")
    t4.round(4).to_csv(os.path.join(a.out, "table4_performance.csv"), index=False, encoding="utf-8-sig")
    iw = pd.DataFrame({m: np.mean(imp_w[m], axis=0) for m in names if imp_w[m]}, index=feats)
    iw.round(2).to_csv(os.path.join(a.out, "feature_weights.csv"), encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = iw[iw.max(axis=1) > 1.5]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(sub) + 2))
        im = ax.imshow(sub.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(sub.shape[1])); ax.set_xticklabels(sub.columns)
        ax.set_yticks(range(sub.shape[0])); ax.set_yticklabels(sub.index)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "feature_weights_heatmap.png"), dpi=150)
    except Exception as e:
        print("히트맵 생략:", e)

    lines = ["[표 2] %d회 반복 AUC" % a.n_runs, t2.round(4).to_string(index=False), "",
             "[표 3] 일원분산분석  F=%.2f  P=%.3g" % (f_stat, p_val), "",
             "[표 4] 평균 성능 (임계값 0.5)", t4.round(3).to_string(index=False), "",
             "[점검] LightGBM AUC 평균 (반복 %d회)" % a.n_runs]
    for k, v in audit.items():
        lines.append("  %-24s %s" % (k, ("%.3f (n=%d회)" % (np.mean(v), len(v))) if v else "-"))
    lines += ["", "해석 가이드: 레이블 섞기 AUC 는 0.5 근처여야 한다. 1일차 AKI 있음/없음의 차이가 크거나, 크레아티닌을 빼면 AUC 가 크게 떨어지면",
              "그 점수는 '예측'이 아니라 이미 진행된 AKI 를 알아본 효과일 가능성이 크다."]
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8").write(txt + "\n")
    print("\n저장:", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build", help="MIMIC-IV 에서 코호트·특징·레이블 표 만들기")
    b.add_argument("--mimic-root", default="team_pipeline/data/mimiciv/3.1_parquet")
    b.add_argument("--out", default="qian_table.parquet")
    b.add_argument("--patients-file", default=None, help="root/hosp/patients 대신 쓸 파일 (.csv.gz 또는 .parquet)")
    b.add_argument("--chartevents-file", default=None, help="root/icu/chartevents 대신 쓸 파일 (.csv.gz 또는 .parquet)")
    b.add_argument("--first-stay-per-patient", action="store_true", help="입원당이 아니라 환자당 첫 입실만 사용")
    b.add_argument("--no-uo", dest="use_uo", action="store_false", help="소변량 기준 레이블을 쓰지 않음 (크레아티닌만)")
    b.set_defaults(func=build, use_uo=True)
    r = sp.add_parser("run", help="균형화 · 6개 모델 · 10회 반복")
    r.add_argument("--table", default="qian_table.parquet")
    r.add_argument("--out", default="qian_out")
    r.add_argument("--n-runs", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--strict", action="store_true", help="결측 보완을 학습 데이터에서만 학습")
    r.add_argument("--exclude-day1-aki", action="store_true", help="1일차 안에 이미 AKI 인 입실 제외")
    r.add_argument("--skip-cnn", action="store_true")
    r.set_defaults(func=run)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
