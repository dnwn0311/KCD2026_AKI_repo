# -*- coding: utf-8 -*-
"""
신장·수분 상태 변수 (팀원 전처리 코드 mimic-iv-aki-preprocessing 의 정의를 따름)

처방 LLM 의 입력(환자 상태 요약)에 아래 정보를 더하기 위한 추출과 계산이다.
  체중, 소변량(최근 6·24시간, mL/kg/h), 24시간 섭취·배설과 수분 균형, 투석(CRRT·혈액투석·복막투석),
  eGFR(CKD-EPI 2021), 요산, 요중 크레아티닌

팀원 코드와 같은 것: 소변·배액·대변·비배설 항목 번호, 투석 항목 번호(225441, 225802, 225805),
  섭취량 규칙(단위 mL, 0 < 양 <= 10000, 시간 겹침 비율로 배분), 소변 기록 간격 규칙(직전 기록과 1초~6시간),
  eGFR 공식(CKD-EPI 2021, 인종 항 없음), 검사 항목(요산 51007, 요중 크레아티닌 51082)
팀원 코드와 다른 것: 체중은 chartevents(41GB)를 다시 읽지 않도록 inputevents 의 patientweight 를 쓴다
  (입실 시 기록 체중이라 팀원 코드의 입실 체중과 대부분 같다). 시점은 입실 48시간 고정이 아니라
  처방 LLM 사례의 시각 t 마다 계산한다.
누출 방지: 시각 t 에 이미 기록된 값만 쓴다 (outputevents·labevents 는 저장 시각, 주입·투석은 시작 시각 기준,
  주입량은 t 이전 부분만).

1) 추출 (한 번):  python prep_kidney.py      -> _work/kidney_*.parquet (환자 단위, PC 밖으로 내보내지 않음)
2) prep_rx.py 가 이 파일들이 있으면 자동으로 프롬프트에 [신장·수분 상태] 줄을 넣는다 (--no-kidney 로 끔)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import common as C

# 팀원 코드 config.yaml 의 itemids
URINE = [226559, 226560, 226561, 226567, 226563, 226564, 226565, 226627, 226631]
URINE_EXCLUDED = [226633, 227489]          # 입원 전 소변, 세척액이 섞인 소변
NOT_OUTPUT = [226626, 226629, 227510, 227511, 227488]   # 수술실 출혈, 경관영양 잔류량, 세척액 주입
DIALYSIS = {225441: "혈액투석", 225802: "CRRT", 225805: "복막투석"}
LAB_EXTRA = {51007: "uric_acid", 51082: "urine_creat"}
WEIGHT_RANGE = (20, 400)
H = np.timedelta64(3600, "s")
CHUNK = 2_000_000


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def find(prefixes):
    """MIMIC_ROOT 아래에서 이름이 prefix 로 시작하는 csv/csv.gz (분할 조각·계획 파일 제외). 표준 위치 우선"""
    hits = []
    base = C.MIMIC.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(C.MIMIC):
        if d.count(os.sep) - base >= 4:
            dirs[:] = []
        for f in files:
            if any(f.startswith(p) for p in prefixes) and f.endswith((".csv", ".csv.gz")) and ".part" not in f:
                hits.append(os.path.join(d, f))
    hits.sort(key=lambda p: (0 if os.sep + "mimiciv" + os.sep in p else 1, len(p)))
    return hits[0] if hits else None


def ts(s):
    return pd.to_datetime(s, errors="coerce").dt.floor("s")


def extract_output(path):
    parts = []
    for ch in pd.read_csv(path, usecols=["stay_id", "charttime", "storetime", "itemid", "value"], chunksize=CHUNK):
        ch = ch[ch["value"].notna() & (ch["value"] >= 0) & ch["stay_id"].notna()]
        ch = ch[~ch["itemid"].isin(URINE_EXCLUDED + NOT_OUTPUT)]
        ct = ts(ch["charttime"])
        parts.append(pd.DataFrame({"stay_id": ch["stay_id"].astype("int64").values, "ct": ct.values,
                                   "tav": ts(ch["storetime"]).fillna(ct).values,
                                   "vol": ch["value"].astype("float32").values,
                                   "urine": ch["itemid"].isin(URINE).values}))
    df = pd.concat(parts, ignore_index=True).dropna(subset=["ct"])
    df = df[(df["vol"] <= 5000)]          # 팀원 코드 허용 범위
    return df


def extract_input(path):
    parts, wparts = [], []
    cols = ["stay_id", "starttime", "endtime", "amount", "amountuom", "patientweight"]
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        ch = ch[ch["stay_id"].notna()]
        st, en = ts(ch["starttime"]), ts(ch["endtime"])
        w = ch["patientweight"]
        wm = w.between(*WEIGHT_RANGE)
        wparts.append(pd.DataFrame({"stay_id": ch.loc[wm, "stay_id"].astype("int64").values,
                                    "wt": st[wm].values, "weight": w[wm].astype("float32").values}))
        m = (ch["amountuom"] == "mL") & (ch["amount"] > 0) & (ch["amount"] <= 10000)
        parts.append(pd.DataFrame({"stay_id": ch.loc[m, "stay_id"].astype("int64").values, "s": st[m].values,
                                   "e": en[m].values, "amount": ch.loc[m, "amount"].astype("float32").values}))
        log("  inputevents 누적 %s행" % f"{sum(len(p) for p in parts):,}")
    intake = pd.concat(parts, ignore_index=True).dropna(subset=["s"])
    wt = pd.concat(wparts, ignore_index=True).dropna(subset=["wt"]).sort_values("wt")
    wt = wt.drop_duplicates("stay_id")    # 체류의 첫 기록 체중 (팀원 코드: 가장 이른 입실 체중)
    return intake, wt


def extract_dialysis(path):
    df = pd.read_csv(path, usecols=["stay_id", "starttime", "endtime", "itemid"])
    df = df[df["itemid"].isin(list(DIALYSIS)) & df["stay_id"].notna()]
    return pd.DataFrame({"stay_id": df["stay_id"].astype("int64").values, "s": ts(df["starttime"]).values,
                         "e": ts(df["endtime"]).values, "itemid": df["itemid"].astype("int32").values}).dropna(subset=["s"])


def extract_labs(path):
    parts, n = [], 0
    for ch in pd.read_csv(path, usecols=["hadm_id", "itemid", "charttime", "storetime", "valuenum"], chunksize=5_000_000,
                          dtype={"hadm_id": "float64", "itemid": "int32", "valuenum": "float32",
                                 "charttime": "string", "storetime": "string"}):
        n += len(ch)
        ch = ch[ch["itemid"].isin(list(LAB_EXTRA)) & ch["hadm_id"].notna() & ch["valuenum"].notna()]
        if len(ch):
            ct = ts(ch["charttime"])
            parts.append(pd.DataFrame({"hadm_id": ch["hadm_id"].astype("int64").values,
                                       "key": ch["itemid"].map(LAB_EXTRA).values,
                                       "t": ts(ch["storetime"]).fillna(ct).values, "value": ch["valuenum"].values}))
        log("  labevents 읽은 행 %s" % f"{n:,}")
    return pd.concat(parts, ignore_index=True).dropna(subset=["t"]) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-labs", action="store_true", help="요산·요중 크레아티닌 생략 (labevents 전체를 읽는 데 15~30분)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(C.WORK, exist_ok=True)
    paths = {"outputevents": find(["outputevents"]), "inputevents": find(["inputevents"]),
             "procedureevents": find(["procedureevents"]), "labevents": C.find_file("labevents", required=False)}
    for k, p in paths.items():
        log("%-16s %s" % (k, p or "없음"))
    miss = [k for k in ("outputevents", "inputevents", "procedureevents") if not paths[k]]
    if miss:
        raise SystemExit("[오류] 파일이 없습니다: %s (MIMIC_ROOT=%s)" % (miss, C.MIMIC))
    summ = {}
    out = extract_output(paths["outputevents"])
    out.to_parquet(os.path.join(C.WORK, "kidney_output.parquet"), index=False)
    summ["output_rows"] = len(out)
    summ["urine_rows"] = int(out["urine"].sum())
    log("outputevents: 배설 기록 %s행 (소변 %s)" % (f"{len(out):,}", f"{summ['urine_rows']:,}"))
    dial = extract_dialysis(paths["procedureevents"])
    dial.to_parquet(os.path.join(C.WORK, "kidney_dialysis.parquet"), index=False)
    summ["dialysis_sessions"] = {DIALYSIS[k]: C.suppress(v) for k, v in dial["itemid"].value_counts().items()}
    log("procedureevents: 투석 기록 %s행" % f"{len(dial):,}")
    intake, wt = extract_input(paths["inputevents"])
    intake.to_parquet(os.path.join(C.WORK, "kidney_intake.parquet"), index=False)
    wt.to_parquet(os.path.join(C.WORK, "kidney_weight.parquet"), index=False)
    summ["intake_rows"], summ["stays_with_weight"] = len(intake), len(wt)
    log("inputevents: 섭취 기록 %s행, 체중 있는 체류 %s건" % (f"{len(intake):,}", f"{len(wt):,}"))
    if not a.skip_labs and paths["labevents"]:
        labs = extract_labs(paths["labevents"])
        if labs is not None:
            labs.to_parquet(os.path.join(C.WORK, "kidney_labs.parquet"), index=False)
            summ["extra_labs"] = {k: C.suppress(v) for k, v in labs["key"].value_counts().items()}
    summ["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(summ, os.path.join(C.RESULTS, "summary_kidney.json"))
    log("완료 (%.1f분). 다음: python prep_rx.py --cap-risk-test 2000 --cap-aki-test 1800" % summ["minutes"])


# ---------------------------------------------------------------------------
# prep_rx.py 에서 쓰는 부분: 사례 시각 t 마다 [신장·수분 상태] 한 줄
def egfr_ckdepi2021(cr, age, female):
    if cr is None or not cr == cr or cr <= 0 or age is None:
        return None
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    v = 142 * min(cr / k, 1) ** a * max(cr / k, 1) ** -1.200 * 0.9938 ** age * (1.012 if female else 1.0)
    return v


class Kidney:
    def __init__(self, stay_ids, hadm_ids):
        def rd(name):
            p = os.path.join(C.WORK, "kidney_%s.parquet" % name)
            return pd.read_parquet(p) if os.path.exists(p) else None
        sids, hids = set(int(x) for x in stay_ids if x == x), set(int(x) for x in hadm_ids)
        out, intake, dial, wt, labs = rd("output"), rd("intake"), rd("dialysis"), rd("weight"), rd("labs")
        self.urine, self.other, self.intake, self.dial, self.labs = {}, {}, {}, {}, {}
        if out is not None:
            out = out[out["stay_id"].isin(sids)].sort_values(["stay_id", "ct"])
            u = out[out["urine"]].copy()
            u = u.groupby(["stay_id", "ct"], as_index=False).agg(tav=("tav", "max"), vol=("vol", "sum"))
            u["gap"] = u.groupby("stay_id")["ct"].diff() / pd.Timedelta(hours=1)
            u = u[(u["gap"] > 1 / 3600) & (u["gap"] <= 6)]   # 팀원 코드: 직전 기록과 1초~6시간
            self.urine = {s: (g["ct"].values, g["tav"].values, g["vol"].values.astype("float64"), g["gap"].values)
                          for s, g in u.groupby("stay_id")}
            o = out[~out["urine"]]
            self.other = {s: (g["ct"].values, g["tav"].values, g["vol"].values.astype("float64"))
                          for s, g in o.groupby("stay_id")}
        if intake is not None:
            intake = intake[intake["stay_id"].isin(sids)]
            self.intake = {s: (g["s"].values, g["e"].values, g["amount"].values.astype("float64"))
                           for s, g in intake.groupby("stay_id")}
        if dial is not None:
            dial = dial[dial["stay_id"].isin(sids)]
            self.dial = {s: (g["s"].values, g["e"].values, g["itemid"].values) for s, g in dial.groupby("stay_id")}
        self.wt = {} if wt is None else {int(s): (w0, float(w)) for s, w0, w in
                                         zip(wt["stay_id"], wt["wt"].values, wt["weight"])}
        if labs is not None:
            labs = labs[labs["hadm_id"].isin(hids)].sort_values("t")
            self.labs = {(h, k): (g["t"].values, g["value"].values) for (h, k), g in labs.groupby(["hadm_id", "key"])}
        self.loaded = out is not None
        self.coverage = []

    @staticmethod
    def _urine(u, t, hours):
        ct, tav, vol, gap = u
        start = t - np.timedelta64(hours * 3600, "s")
        st = ct - (gap * 3600).astype("int64").astype("timedelta64[s]")
        m = (tav <= t) & (ct > start) & (st < t)
        if not m.any():
            return 0.0, 0.0
        ov = (np.minimum(ct[m], t) - np.maximum(st[m], start)) / H
        ov = np.clip(ov, 0, None)
        return float(np.sum(vol[m] * ov / gap[m])), float(np.sum(ov))

    def _lab(self, hadm, key, t, hours=72):
        v = self.labs.get((hadm, key))
        if v is None:
            return None
        tt, vv = v
        m = (tt <= t) & (tt >= t - np.timedelta64(hours * 3600, "s"))
        return float(vv[m][-1]) if m.any() else None

    def line(self, row, cr_latest):
        """반환: (프롬프트 한 줄, 집계용 dict)"""
        t = np.datetime64(row["t"])
        sid = int(row["stay_id"]) if row.get("stay_id") == row.get("stay_id") and row.get("stay_id") is not None else None
        parts, cov = [], {}
        w = self.wt.get(sid) if sid is not None else None
        weight = w[1] if w is not None and w[0] <= t else None
        cov["weight"] = weight is not None
        if weight:
            parts.append("체중 %.0fkg" % weight)
        eg = egfr_ckdepi2021(cr_latest, float(row["age"]), row["gender"] == "F")
        cov["egfr"] = eg is not None
        if eg is not None:
            parts.append("eGFR(CKD-EPI 2021) %.0f" % eg)
        u = self.urine.get(sid)
        cov["urine"] = False
        if u is not None:
            for hrs in (6, 24):
                vol, cover = self._urine(u, t, hrs)
                if cover >= hrs * 0.5:
                    cov["urine"] = True
                    if weight:
                        parts.append("소변량 최근 %d시간 %.2f mL/kg/h (%s mL)" % (hrs, vol / weight / cover, f"{vol:,.0f}"))
                    else:
                        parts.append("소변량 최근 %d시간 %s mL (기록 %.0f시간)" % (hrs, f"{vol:,.0f}", cover))
                elif hrs == 6:
                    parts.append("소변량 최근 6시간 기록 부족")
        d24 = t - np.timedelta64(24 * 3600, "s")
        intake = None
        g = self.intake.get(sid)
        if g is not None:
            s, e, amt = g
            m = (s <= t) & ((e > d24) | ((e != e) & (s > d24)))
            if m.any():
                ss, ee, aa = s[m], e[m], amt[m]
                dur = (ee - ss) / H
                ov = (np.minimum(ee, t) - np.maximum(ss, d24)) / H
                frac = np.where(dur > 0, np.clip(ov, 0, None) / np.where(dur > 0, dur, 1), (ss > d24).astype(float))
                intake = float(np.sum(aa * frac))
        cov["intake"] = intake is not None
        uvol24, ucov24 = self._urine(u, t, 24) if u is not None else (0.0, 0.0)
        o = self.other.get(sid)
        other24 = 0.0
        if o is not None:
            oct_, otav, ovol = o
            other24 = float(np.sum(ovol[(otav <= t) & (oct_ > d24) & (oct_ <= t)]))
        if intake is not None:
            out_txt = "24시간 섭취 %s mL" % f"{intake:,.0f}"
            if ucov24 >= 12:
                bal = intake - uvol24 - other24
                out_txt += ", 배설 %s mL (소변 %s, 기타 %s), 수분 균형 %+.0f mL" % (
                    f"{uvol24 + other24:,.0f}", f"{uvol24:,.0f}", f"{other24:,.0f}", bal)
            parts.append(out_txt)
        dz = self.dial.get(sid)
        dtxt = "없음"
        if dz is not None:
            s, e, it = dz
            on = (s <= t) & ((e > t) | (e != e))
            rec = (s <= t) & (e > d24) & (e <= t)
            ever = s <= t
            if on.any():
                dtxt = "%s 진행 중" % DIALYSIS[int(it[on][0])]
            elif rec.any():
                dtxt = "최근 24시간 %s" % DIALYSIS[int(it[rec][-1])]
            elif ever.any():
                dtxt = "이번 체류 중 %s 이력" % DIALYSIS[int(it[ever][-1])]
        cov["dialysis"] = dtxt != "없음"
        parts.append("투석 " + dtxt)
        for key, name, unit in (("uric_acid", "요산", "mg/dL"), ("urine_creat", "요중 크레아티닌", "mg/dL")):
            v = self._lab(int(row["hadm_id"]), key, t)
            if v is not None:
                parts.append("%s %.1f %s" % (name, v, unit))
        return "[신장·수분 상태] " + ", ".join(parts), cov


if __name__ == "__main__":
    main()
