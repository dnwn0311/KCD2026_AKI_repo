# -*- coding: utf-8 -*-
"""
1단계: 원본에서 필요한 부분만 추출해 _work/ 에 parquet 로 저장

- labevents   : 정해 둔 검사 항목, 입원 번호가 있는 행만
- chartevents : 정해 둔 활력징후 항목만 (화씨 체온은 섭씨로 변환, 측정 오류 범위 제외)
                중환자실 체류 기간(icu_stays): icustays 파일이 있으면 그것을, 없으면 stay_id 별 첫·마지막 기록 시각
- prescriptions: 주 약물(MAIN)과 첨가 약물(ADDITIVE)만, 약물명 정규화, 1회 용량·단위·하루 투여 횟수
- poe          : 투약 오더의 중단(D/C)·변경(Change) 시각을 처방에 연결 (실제 종료 시각 = 예정 종료와 중단 중 빠른 쪽)
- patients, admissions

시각은 결과가 등록된 시각(storetime)을 우선 쓰고, 없으면 charttime 을 씁니다.
화면과 results/ 에는 건수 같은 집계만 출력합니다.

사용:  python prep_extract.py [--only labs vitals rx] [--chunksize 5000000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import common as C

TS_FMT = "%Y-%m-%d %H:%M:%S"


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def to_ts(s):
    return pd.to_datetime(s, format=TS_FMT, errors="coerce")


def check_columns(path, needed):
    head = pd.read_csv(path, nrows=0)
    missing = [c for c in needed if c not in head.columns]
    if missing:
        raise SystemExit("[오류] %s 에 필요한 컬럼이 없습니다: %s" % (os.path.basename(path), missing))


def validate_items(dict_path, id_col, label_col, groups, kind):
    """사전(d_labitems/d_items)의 이름과 itemid 를 대조. 맞지 않는 itemid 는 제외한다."""
    report = {}
    if not dict_path:
        log("  %s 사전이 없어 itemid 이름 대조를 건너뜁니다." % kind)
        return groups, {"skipped": True}
    d = pd.read_csv(dict_path, usecols=[id_col, label_col])
    names = dict(zip(d[id_col].astype(int), d[label_col].astype(str).str.lower()))
    out = {}
    for key, (ids, word) in groups.items():
        keep = []
        for i in ids:
            nm = names.get(int(i))
            ok = nm is not None and word in nm
            report["%s:%d" % (key, i)] = "%s (%s)" % ("OK" if ok else "제외", nm)
            if ok:
                keep.append(int(i))
        out[key] = (keep, word)
    return out, report


def extract_labs(chunksize, admit):
    path = C.find_file("labevents")
    check_columns(path, ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"])
    groups = {k: (C.LAB_INFO[k][5], C.LAB_INFO[k][6]) for k in C.LAB_KEYS}
    groups, rep = validate_items(C.find_file("d_labitems", required=False), "itemid", "label", groups, "검사")
    id2code = {}
    for code, k in enumerate(C.LAB_KEYS):
        for i in groups[k][0]:
            id2code[i] = code
    ids = np.array(sorted(id2code))
    log("labevents 추출 시작: 항목 %d개 itemid" % len(ids))
    parts, rows_in = [], 0
    reader = pd.read_csv(path, usecols=["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"],
                         dtype={"subject_id": "int64", "hadm_id": "float64", "itemid": "int32",
                                "valuenum": "float32", "charttime": "string", "storetime": "string"},
                         chunksize=chunksize)
    for ch in reader:
        rows_in += len(ch)
        ch = ch[ch["itemid"].isin(ids) & ch["hadm_id"].notna() & ch["valuenum"].notna()]
        if len(ch) == 0:
            continue
        t = to_ts(ch["storetime"]).fillna(to_ts(ch["charttime"]))
        out = pd.DataFrame({
            "subject_id": ch["subject_id"].astype("int64").values,
            "hadm_id": ch["hadm_id"].astype("int64").values,
            "code": ch["itemid"].map(id2code).astype("int8").values,
            "value": ch["valuenum"].astype("float32").values,
            "t": t.values,
        }).dropna(subset=["t"])
        parts.append(out)
        log("  읽은 행 %s, 추출 누적 %s" % (f"{rows_in:,}", f"{sum(len(p) for p in parts):,}"))
    df = pd.concat(parts, ignore_index=True)
    df = keep_in_admission(df, admit)
    df.to_parquet(os.path.join(C.WORK, "labs.parquet"), index=False)
    log("labevents 완료: 입원 기간 내 %s행" % f"{len(df):,}")
    return {"rows_read": rows_in, "rows_kept": len(df), "itemid_check": rep,
            "by_lab": {C.LAB_KEYS[c]: C.suppress(n) for c, n in df["code"].value_counts().items()}}


def extract_vitals(chunksize, admit):
    path = C.find_file("chartevents", required=False)
    if not path:
        log("chartevents 가 없어 활력징후를 건너뜁니다.")
        return {"skipped": True}
    size = os.path.getsize(path)
    if path.endswith("chartevents.csv-001-013.gz") and size != C.CHARTEVENTS_EXPECTED_BYTES:
        raise SystemExit("[오류] chartevents 사본의 크기가 서버 파일과 다릅니다 (%d bytes)." % size)
    check_columns(path, ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"])
    groups = {k: (C.VITAL_INFO[k][3], C.VITAL_INFO[k][4]) for k in C.VITAL_KEYS}
    groups, rep = validate_items(C.find_file("d_items", required=False), "itemid", "label", groups, "활력징후")
    id2code = {}
    for code, k in enumerate(C.VITAL_KEYS):
        for i in groups[k][0]:
            id2code[i] = code
    ids = np.array(sorted(id2code))
    lo = np.array([C.VITAL_INFO[k][5] for k in C.VITAL_KEYS], dtype="float32")
    hi = np.array([C.VITAL_INFO[k][6] for k in C.VITAL_KEYS], dtype="float32")
    log("chartevents 추출 시작 (%s): itemid %d개" % (os.path.basename(path), len(ids)))
    parts, rows_in, stay_parts = [], 0, []
    reader = pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime",
                                        "valuenum"],
                         dtype={"subject_id": "int64", "hadm_id": "float64", "stay_id": "float64", "itemid": "int32",
                                "valuenum": "float32", "charttime": "string", "storetime": "string"},
                         chunksize=chunksize)
    for ch in reader:
        rows_in += len(ch)
        ch = ch[ch["itemid"].isin(ids) & ch["hadm_id"].notna() & ch["valuenum"].notna()]
        if len(ch) == 0:
            continue
        # 중환자실 체류 기간 (icustays 파일이 없을 때 사용): stay_id 별 첫·마지막 활력징후 기록 시각
        sv = pd.DataFrame({"stay_id": ch["stay_id"].values, "hadm_id": ch["hadm_id"].values,
                           "ct": to_ts(ch["charttime"]).values}).dropna()
        if len(sv):
            stay_parts.append(sv.groupby("stay_id").agg(hadm_id=("hadm_id", "first"), intime=("ct", "min"),
                                                        outtime=("ct", "max")).reset_index())
        val = ch["valuenum"].astype("float32").values.copy()
        isf = (ch["itemid"].values == C.TEMP_F_ITEMID)
        val[isf] = (val[isf] - 32.0) * 5.0 / 9.0
        code = ch["itemid"].map(id2code).astype("int8").values
        ok = (val >= lo[code]) & (val <= hi[code])
        t = to_ts(ch["storetime"]).fillna(to_ts(ch["charttime"]))
        out = pd.DataFrame({
            "subject_id": ch["subject_id"].astype("int64").values,
            "hadm_id": ch["hadm_id"].astype("int64").values,
            "code": code, "value": val, "t": t.values,
        })[ok].dropna(subset=["t"])
        parts.append(out)
        log("  읽은 행 %s, 추출 누적 %s" % (f"{rows_in:,}", f"{sum(len(p) for p in parts):,}"))
    df = pd.concat(parts, ignore_index=True)
    df = keep_in_admission(df, admit)
    df.to_parquet(os.path.join(C.WORK, "vitals.parquet"), index=False)
    log("chartevents 완료: 입원 기간 내 %s행" % f"{len(df):,}")
    stays = pd.concat(stay_parts, ignore_index=True).groupby("stay_id").agg(
        hadm_id=("hadm_id", "first"), intime=("intime", "min"), outtime=("outtime", "max")).reset_index()
    stays.to_parquet(os.path.join(C.WORK, "icu_stays_from_chartevents.parquet"), index=False)
    return {"file": os.path.basename(path), "rows_read": rows_in, "rows_kept": len(df), "itemid_check": rep,
            "by_vital": {C.VITAL_KEYS[c]: C.suppress(n) for c, n in df["code"].value_counts().items()}}


def build_icu_stays():
    """중환자실 체류 기간: icustays 파일 우선, 없으면 chartevents 에서 만든 근사값"""
    path = C.find_file("icustays", required=False)
    if path:
        st = pd.read_csv(path, usecols=["stay_id", "hadm_id", "intime", "outtime"])
        st["intime"], st["outtime"] = to_ts(st["intime"]), to_ts(st["outtime"])
        source = os.path.basename(path)
    else:
        p = os.path.join(C.WORK, "icu_stays_from_chartevents.parquet")
        if not os.path.exists(p):
            log("중환자실 체류 기간을 만들 수 없습니다 (icustays 파일도, chartevents 추출 결과도 없음).")
            return {"skipped": True}
        st = pd.read_parquet(p)
        source = "chartevents 첫·마지막 활력징후 기록 (icustays 파일 없음)"
    st = st.dropna(subset=["hadm_id", "intime", "outtime"])
    st = st[st["outtime"] > st["intime"]].copy()
    st["hadm_id"] = st["hadm_id"].astype("int64")
    st.to_parquet(os.path.join(C.WORK, "icu_stays.parquet"), index=False)
    hours = (st["outtime"] - st["intime"]) / pd.Timedelta(hours=1)
    log("중환자실 체류 %s건 (출처: %s)" % (f"{len(st):,}", source))
    return {"source": source, "stays": C.suppress(len(st)), "admissions_with_icu": C.suppress(st["hadm_id"].nunique()),
            "median_hours": round(float(hours.median()), 1)}


def poe_key(s):
    """poe_id ("환자번호-순번") -> 정수 키. 없으면 -1"""
    s = s.astype("string")
    parts = s.str.split("-", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(-1, index=s.index, dtype="int64")
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    k = a * 1_000_000 + b
    return k.fillna(-1).astype("int64")


def extract_poe_dc(chunksize):
    """투약 오더별 중단(D/C)·변경(Change) 시각. 변경 오더도 원래 오더를 끝내므로 함께 기록"""
    path = C.find_file("poe", required=False)
    if not path:
        log("poe 파일이 없어 중단 정답을 만들 수 없습니다 (신규·감량·증량만 사용).")
        return None, {"skipped": True}
    need = ["ordertime", "order_type", "transaction_type", "discontinue_of_poe_id"]
    check_columns(path, need)
    log("poe 추출 시작 (중단·변경 오더)")
    parts, rows_in = [], 0
    for ch in pd.read_csv(path, usecols=need, dtype="string", chunksize=chunksize):
        rows_in += len(ch)
        ch = ch[(ch["order_type"] == "Medications") & ch["discontinue_of_poe_id"].notna()
                & ch["transaction_type"].isin(["D/C", "Change"])]
        if len(ch) == 0:
            continue
        parts.append(pd.DataFrame({
            "key": poe_key(ch["discontinue_of_poe_id"]).values,
            "dc_time": to_ts(ch["ordertime"]).values,
            "dc_type": np.where(ch["transaction_type"].values == "D/C", 1, 2).astype("int8"),
        }))
        log("  읽은 행 %s" % f"{rows_in:,}")
    dc = pd.concat(parts, ignore_index=True).dropna(subset=["dc_time"])
    dc = dc[dc["key"] >= 0].sort_values("dc_time").drop_duplicates("key", keep="first")
    rep = {"rows_read": rows_in, "dc_orders": C.suppress((dc["dc_type"] == 1).sum()),
           "change_orders": C.suppress((dc["dc_type"] == 2).sum())}
    return dc.set_index("key"), rep


def extract_rx(chunksize, admit):
    path = C.find_file("prescriptions")
    need = ["subject_id", "hadm_id", "poe_id", "starttime", "stoptime", "drug_type", "drug",
            "dose_val_rx", "dose_unit_rx", "doses_per_24_hrs"]
    check_columns(path, need)
    dc, dc_rep = extract_poe_dc(chunksize)
    log("prescriptions 추출 시작")
    parts, rows_in = [], 0
    cache = {}
    for ch in pd.read_csv(path, usecols=need, dtype="string", chunksize=chunksize):
        rows_in += len(ch)
        ch = ch[ch["drug_type"].isin(["MAIN", "ADDITIVE"]) & ch["hadm_id"].notna() & ch["starttime"].notna()]
        names = []
        for d in ch["drug"].fillna("").tolist():
            if d not in cache:
                cache[d] = C.normalize_drug(d)
            names.append(cache[d])
        st = to_ts(ch["starttime"])
        sp = to_ts(ch["stoptime"])
        out = pd.DataFrame({
            "subject_id": ch["subject_id"].astype("int64").values,
            "hadm_id": ch["hadm_id"].astype("float64").astype("int64").values,
            "drug": names, "start": st.values, "stop": sp.values,
            "dose": pd.to_numeric(ch["dose_val_rx"], errors="coerce").astype("float32").values,
            "unit": ch["dose_unit_rx"].fillna("").str.strip().str.lower().values,
            "per24": pd.to_numeric(ch["doses_per_24_hrs"], errors="coerce").astype("float32").values,
            "key": poe_key(ch["poe_id"]).values,
        })
        out = out[out["drug"].notna() & out["start"].notna()]
        parts.append(out)
        log("  읽은 행 %s" % f"{rows_in:,}")
    df = pd.concat(parts, ignore_index=True)
    df["stop"] = df["stop"].fillna(df["start"] + pd.Timedelta(hours=24))
    df = df[df["hadm_id"].isin(admit.index)]
    if dc is not None:
        df = df.join(dc, on="key")
        ok = df["dc_time"].notna() & (df["dc_time"] >= df["start"])
        df.loc[~ok, ["dc_time"]] = pd.NaT
        df["dc_type"] = df["dc_type"].where(ok, 0).fillna(0).astype("int8")
        df["stop"] = df["stop"].where(~ok | (df["stop"] <= df["dc_time"]), df["dc_time"])  # 실제 종료 시각
    else:
        df["dc_time"], df["dc_type"] = pd.NaT, np.int8(0)
    df = df.drop(columns=["key"])
    df["unit"] = df["unit"].astype("category")
    df.to_parquet(os.path.join(C.WORK, "rx.parquet"), index=False)
    log("prescriptions 완료: %s행, 정규화 약물명 %d종" % (f"{len(df):,}", df["drug"].nunique()))
    return {"rows_read": rows_in, "rows_kept": len(df), "distinct_drugs": int(df["drug"].nunique()),
            "distinct_raw_names": len(cache), "dose_numeric_rate": round(float(df["dose"].notna().mean()), 3),
            "linked_dc": C.suppress((df["dc_type"] == 1).sum()), "linked_change": C.suppress((df["dc_type"] == 2).sum()),
            "poe": dc_rep}


def load_base():
    p = pd.read_csv(C.find_file("patients"), usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    a = pd.read_csv(C.find_file("admissions"),
                    usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                             "admission_location"])
    a["admittime"] = to_ts(a["admittime"])
    a["dischtime"] = to_ts(a["dischtime"])
    a = a.dropna(subset=["admittime", "dischtime"])
    a = a[a["dischtime"] > a["admittime"]]
    p.to_parquet(os.path.join(C.WORK, "patients.parquet"), index=False)
    a.to_parquet(os.path.join(C.WORK, "admissions.parquet"), index=False)
    return p, a


def keep_in_admission(df, admit):
    m = df.join(admit[["admittime", "dischtime"]], on="hadm_id", how="inner")
    m = m[(m["t"] >= m["admittime"]) & (m["t"] <= m["dischtime"])]
    return m.drop(columns=["admittime", "dischtime"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["labs", "vitals", "rx"])
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    a = ap.parse_args()
    os.makedirs(C.WORK, exist_ok=True)
    t0 = time.time()
    summary = {"mimic_root": C.MIMIC}
    _, adm = load_base()
    admit = adm.set_index("hadm_id")
    summary["admissions"] = len(adm)
    log("입원 %s건" % f"{len(adm):,}")
    if "labs" in a.only:
        summary["labs"] = extract_labs(a.chunksize, admit)
    if "vitals" in a.only:
        summary["vitals"] = extract_vitals(a.chunksize, admit)
    summary["icu_stays"] = build_icu_stays()
    if "rx" in a.only:
        summary["prescriptions"] = extract_rx(a.chunksize, admit)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(summary, os.path.join(C.RESULTS, "summary_extract.json"))
    log("1단계 완료 (%.1f분). 집계: results/summary_extract.json" % summary["minutes"])


if __name__ == "__main__":
    main()
