# -*- coding: utf-8 -*-
"""
2단계 (AKI): 중환자실 입실 후 48시간 이후를 대상으로 두 가지 학습 데이터를 만든다

AKI 정의 (KDIGO 크레아티닌 기준, 소변량 기준은 outputevents 가 없어 제외)
  - 48시간 안의 최저값보다 0.3 mg/dL 이상 상승, 또는
  - 7일 안의 최저값(기저치)의 1.5배 이상
  크레아티닌은 결과가 등록된 시각 기준 (의사가 실제로 볼 수 있는 시각)

대상: 성인(18세 이상)의 입원별 첫 중환자실 체류

A. 위험 예측 데이터 (data/aki_risk.parquet, LightGBM 용) - 이동 창 방식
  - 예측 시점 T = 입실 후 48, 72, 96 … 시간 (24시간마다, 체류당 최대 14개). T 에 중환자실에 있어야 함
  - 입력: T 직전 48시간의 검사·활력징후 요약, T 시점 투여 약과 직전 48시간에 시작된 약,
          크레아티닌 기저치(7일 안 최저값)와 최근 추이, 나이·성별·입원 정보, 입실 후 경과 시간
  - 정답: T 이후 24시간 안에 새로 AKI 가 생겼는지
  - 제외: 첫 AKI 이후의 예측 시점(이미 AKI), T 전 7일 안에 크레아티닌이 없는 경우, 이후 24시간 안에 크레아티닌을
          재지 않은 경우(정답을 정할 수 없음)
  - 한 환자에게서 여러 시점이 나오므로 분할은 환자 단위, weight 열(체류당 1/시점 수)로 장기 체류 쏠림을 보정

B. 처방 변경 데이터 (Qwen 학습용 JSONL) 는 위험 예측 모델을 학습한 뒤 prep_rx.py 에서 만든다
  (예측된 위험도를 LLM 입력에 넣어야 하므로). 이 파일에는 공통 함수(aki_events, rx_samples 등)가 있다.

환자 단위로 train 80 / val 10 / test 10. results/summary_aki.json 에는 집계만 (10 미만은 <10).

사용:  python prep_aki.py
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as C
from prep_build import asof_values, fmt_dose, fmt_num, load, log

H = pd.Timedelta(hours=1)
CR = C.LAB_KEYS.index("creatinine")


# ---------------------------------------------------------------------------
def first_stays(stays, adm, pat):
    st = stays.sort_values("intime").drop_duplicates("hadm_id")
    st = st.merge(adm[["hadm_id", "subject_id", "admittime", "dischtime", "admission_type", "admission_location"]],
                  on="hadm_id", how="inner")
    st = st.merge(pat, on="subject_id", how="inner")
    st["age"] = st["anchor_age"] + (st["intime"].dt.year - st["anchor_year"])
    st = st[st["age"] >= 18].copy()
    st[["admission_type", "admission_location"]] = st[["admission_type", "admission_location"]].fillna("미상")
    st["T"] = st["intime"] + C.AKI_ICU_HOURS * H  # 처방 모델: 이 시각 이후의 AKI 만 사용
    st["split"] = st["subject_id"].map(C.subject_split)
    return st.reset_index(drop=True)


def creatinine_flags(labs, st):
    """입원별 크레아티닌마다 48시간 최저값, 7일 최저값(기저치), AKI 여부"""
    cr = labs.loc[(labs["code"] == CR) & labs["hadm_id"].isin(st["hadm_id"]), ["hadm_id", "t", "value"]]
    cr = cr[(cr["value"] > 0) & (cr["value"] < 30)].sort_values(["hadm_id", "t"]).reset_index(drop=True)
    g = cr.set_index("t").groupby("hadm_id", sort=True)["value"]
    cr["min48"] = g.rolling("%dh" % C.AKI_ABS_HOURS, closed="left").min().values
    cr["base"] = g.rolling("%dD" % C.AKI_REL_DAYS, closed="left").min().values
    cr["aki"] = ((cr["value"] - cr["min48"]) >= C.AKI_ABS_RISE) | ((cr["value"] / cr["base"]) >= C.AKI_REL_RISE)
    return cr


# ---------------------------------------------------------------------------
def explode_points(df, pts, lo, hi, hcol=None, hi_open=False, lo_open=False):
    """각 행을 '예측 시점 기준 [T+lo, T+hi]' 범위에 들어가는 예측 시점(k)마다 복제한다.

    df 는 hadm_id 와 시각 열 t(또는 hcol 로 준 입실 후 시간)를 가진다.
    pts: hadm_id 별 intime, n_points. 반환: df 의 해당 행 + k, pid(=hadm_id*1000+k), T(입실 후 시간)
    """
    F, S = C.RISK_FIRST_HOURS, C.RISK_STEP_HOURS
    info = pts.set_index("hadm_id")
    df = df[df["hadm_id"].isin(info.index)]
    if hcol is None:
        h = ((df["t"] - df["hadm_id"].map(info["intime"])) / H).values
    else:
        h = df[hcol].values
    eps = 1e-6
    kl = np.ceil((h - F - hi + (eps if hi_open else 0)) / S)
    kh = np.floor((h - F - lo - (eps if lo_open else 0)) / S)
    kl = np.maximum(kl, 0)
    kh = np.minimum(kh, df["hadm_id"].map(info["n_points"]).values - 1)
    n = np.where(np.isfinite(kl) & np.isfinite(kh), kh - kl + 1, 0).astype("int64")
    n = np.maximum(n, 0)
    idx = np.repeat(np.arange(len(df)), n)
    off = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)
    out = df.iloc[idx].copy()
    out["k"] = (kl[idx] + off).astype("int64")
    out["pid"] = out["hadm_id"].values.astype("int64") * 1000 + out["k"].values
    out["Th"] = F + S * out["k"].values
    out["h"] = h[idx]
    return out


def agg_window(tab, keys, pts, prefix):
    """예측 시점마다 직전 48시간의 항목별 최신·최저·최고·평균"""
    m = explode_points(tab[["hadm_id", "code", "t", "value"]], pts, -C.RISK_WINDOW_HOURS, 0).sort_values("t")
    g = m.groupby(["pid", "code"])["value"]
    a = pd.DataFrame({"last": g.last(), "min": g.min(), "max": g.max(), "mean": g.mean()}).unstack("code")
    a.columns = ["%s%s_%s" % (prefix, keys[c], stat) for stat, c in a.columns]
    return a, m


def med_features(rx, pts):
    """T 시점에 투여 중인 약, 직전 48시간에 시작된 약을 AKI 분류별로 센다"""
    F, S = C.RISK_FIRST_HOURS, C.RISK_STEP_HOURS
    info = pts.set_index("hadm_id")
    r = rx[rx["hadm_id"].isin(info.index)].copy()
    r["sh"] = ((r["start"] - r["hadm_id"].map(info["intime"])) / H).values
    r["eh"] = ((r["stop"] - r["hadm_id"].map(info["intime"])) / H).values
    # 투여 중: start <= T < stop  ->  k 범위 [ceil((sh-F)/S), ceil((eh-F)/S)-1]
    kl = np.maximum(np.ceil((r["sh"].values - F) / S), 0)
    kh = np.minimum(np.ceil((r["eh"].values - F) / S) - 1, r["hadm_id"].map(info["n_points"]).values - 1)
    n = np.maximum(np.where(np.isfinite(kl) & np.isfinite(kh), kh - kl + 1, 0), 0).astype("int64")
    idx = np.repeat(np.arange(len(r)), n)
    off = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)
    act = r.iloc[idx].copy()
    act["pid"] = act["hadm_id"].values.astype("int64") * 1000 + (kl[idx] + off).astype("int64")
    started = explode_points(r, pts, -C.RISK_WINDOW_HOURS, 0, hcol="sh")
    out = pd.DataFrame(index=pd.Index(pts_pids(pts), name="pid"))
    out["med_active_n"] = act.groupby("pid")["drug"].nunique()
    out["med_started_n"] = started.groupby("pid")["drug"].nunique()
    for cat, _ in C.AKI_DRUG_GROUPS:
        out["med_active_%s" % cat] = act[act["cat"] == cat].groupby("pid")["drug"].nunique()
        out["med_started_%s" % cat] = started[started["cat"] == cat].groupby("pid")["drug"].nunique()
    flags = {"vancomycin": ["vancomycin"], "piptazo": ["piperacillin-tazobactam"],
             "nsaid": ["ketorolac", "ibuprofen", "naproxen", "celecoxib", "diclofenac", "indomethacin", "meloxicam"],
             "aminoglycoside": ["gentamicin", "tobramycin", "amikacin"], "loop": ["furosemide", "bumetanide", "torsemide"],
             "vasopressor": C.VASOPRESSORS}
    for k, words in flags.items():
        hit = act[act["drug"].str.contains("|".join(words), regex=True)]
        out["med_active_%s" % k] = hit.groupby("pid").size().gt(0).astype("int8")
    out = out.fillna(0)
    out["med_vanc_piptazo"] = ((out["med_active_vancomycin"] > 0) & (out["med_active_piptazo"] > 0)).astype("int8")
    return out


def pts_pids(pts):
    return np.concatenate([h * 1000 + np.arange(n) for h, n in zip(pts["hadm_id"].values.astype("int64"),
                                                                  pts["n_points"].values)]) \
        if len(pts) else np.array([], "int64")


def build_risk(st, cr, labs, vitals, rx):
    F, S, W, HZ = C.RISK_FIRST_HOURS, C.RISK_STEP_HOURS, C.RISK_WINDOW_HOURS, C.RISK_HORIZON
    pts = st.copy()
    # 예측 시점: T 에 중환자실에 있고, T+24시간까지 입원 중
    last_h = np.minimum((pts["outtime"] - pts["intime"]) / H, (pts["dischtime"] - pts["intime"]) / H - HZ)
    pts["n_points"] = np.clip(np.floor((last_h - F) / S) + 1, 0, C.RISK_MAX_POINTS).astype("int64")
    pts = pts[pts["n_points"] > 0].reset_index(drop=True)
    n_stays, n_cand = len(pts), int(pts["n_points"].sum())
    log("위험 예측 후보: 체류 %s건, 예측 시점 %s개" % (f"{n_stays:,}", f"{n_cand:,}"))

    P = pd.DataFrame({"pid": pts_pids(pts)})
    P["hadm_id"] = P["pid"] // 1000
    P["k"] = P["pid"] % 1000
    P["Th"] = F + S * P["k"]
    P = P.merge(pts[["hadm_id", "subject_id", "intime", "admittime", "age", "gender", "admission_type",
                     "admission_location", "split"]], on="hadm_id", how="left")
    P["T"] = P["intime"] + P["Th"] * H

    # 첫 AKI (입실 48시간 전부터) 이후의 시점은 제외
    c = cr[cr["hadm_id"].isin(pts["hadm_id"])].merge(pts[["hadm_id", "intime"]], on="hadm_id")
    c["h"] = (c["t"] - c["intime"]) / H
    first = c[c["aki"] & (c["h"] >= -48)].groupby("hadm_id")["h"].min()
    P["first_aki_h"] = P["hadm_id"].map(first)
    after_aki = P["first_aki_h"].notna() & (P["Th"] >= P["first_aki_h"])

    # 크레아티닌: 기저치용 7일 창, 정답용 이후 24시간
    cw = explode_points(c, pts, -C.AKI_REL_DAYS * 24, 0, hcol="h").sort_values("t")
    cz = explode_points(c, pts, 0, HZ, hcol="h", lo_open=True)
    has_cr = P["pid"].isin(set(cw["pid"]))
    measured = P["pid"].isin(set(cz["pid"]))
    label = set(cz.loc[cz["aki"], "pid"])
    keep = ~after_aki & has_cr & measured
    excl = {"stays_with_points": C.suppress(n_stays), "candidate_points": C.suppress(n_cand),
            "excluded_after_first_aki": C.suppress(int(after_aki.sum())),
            "excluded_no_creatinine_7d": C.suppress(int((~after_aki & ~has_cr).sum())),
            "excluded_no_creatinine_in_horizon": C.suppress(int((~after_aki & has_cr & ~measured).sum()))}
    P = P[keep].copy()
    P["label"] = P["pid"].isin(label).astype("int8")
    P["weight"] = 1.0 / P.groupby("hadm_id")["pid"].transform("count")
    pts = pts[pts["hadm_id"].isin(P["hadm_id"])]
    log("위험 예측 표본 %s개 (체류 %s건)" % (f"{len(P):,}", f"{P['hadm_id'].nunique():,}"))

    # 크레아티닌 특징
    cw = cw[cw["pid"].isin(P["pid"])]
    g = cw.groupby("pid")
    w48 = cw[cw["h"] >= cw["Th"] - W].groupby("pid")
    f = pd.DataFrame(index=pd.Index(P["pid"], name="pid"))
    f["cr_last"] = g["value"].last()
    f["cr_base7d"] = g["value"].min()
    f["cr_hours_since_last"] = (cw.groupby("pid")["Th"].last() - g["h"].last())
    f["cr_first48h"] = w48["value"].first()
    f["cr_min48h"] = w48["value"].min()
    f["cr_max48h"] = w48["value"].max()
    f["cr_n48h"] = w48.size()
    f["cr_last_minus_base"] = f["cr_last"] - f["cr_base7d"]
    f["cr_last_over_base"] = f["cr_last"] / f["cr_base7d"]
    f["cr_last_minus_min48h"] = f["cr_last"] - f["cr_min48h"]
    f["cr_change48h"] = f["cr_last"] - f["cr_first48h"]
    f["cr_n48h"] = f["cr_n48h"].fillna(0)

    # 직전 48시간 검사·활력징후, 약
    lab_f, _ = agg_window(labs[labs["hadm_id"].isin(pts["hadm_id"])], C.LAB_KEYS, pts, "lab_")
    vit_f = pd.DataFrame(index=f.index)
    if vitals is not None:
        vit_f, mv = agg_window(vitals[vitals["hadm_id"].isin(pts["hadm_id"])], C.VITAL_KEYS, pts, "vit_")
        vit_f["vit_map_lt65_n"] = mv[(mv["code"] == C.VITAL_KEYS.index("map")) & (mv["value"] < 65)].groupby("pid").size()
        vit_f["vit_hr_gt100_n"] = mv[(mv["code"] == C.VITAL_KEYS.index("heart_rate")) & (mv["value"] > 100)].groupby("pid").size()
    med_f = med_features(rx[rx["hadm_id"].isin(pts["hadm_id"])], pts)

    base = P.set_index("pid")
    out = pd.DataFrame(index=f.index)
    out["age"] = base["age"]
    out["male"] = (base["gender"] == "M").astype("int8")
    out["admission_type"] = base["admission_type"].astype("category")
    out["admission_location"] = base["admission_location"].astype("category")
    out["icu_hours_at_T"] = base["Th"].astype("float32")
    out["hosp_hours_at_T"] = (base["T"] - base["admittime"]) / H
    out = out.join(f).join(lab_f).join(vit_f).join(med_f)
    for col in ("vit_map_lt65_n", "vit_hr_gt100_n"):
        if col in out:
            out[col] = out[col].fillna(0)
    out["label"] = base["label"]
    out["weight"] = base["weight"]
    out["split"] = base["split"]
    out["stay_key"] = base["hadm_id"]  # 체류 구분용 (분할·집계에만 사용, 모델 입력 아님)
    out["pid"] = out.index.values      # 예측 시점 번호 (= 입원번호*1000 + 순번, 모델 입력 아님)
    excl["points_per_stay_median"] = float(P.groupby("hadm_id").size().median())
    excl["positive_rate_by_icu_day"] = {
        "%d일차" % (int(d) // 24 + 1): round(float(v), 4)
        for d, v in P.groupby("Th")["label"].mean().items() if (P["Th"] == d).sum() >= 10}
    return out.reset_index(drop=True), excl


# ---------------------------------------------------------------------------
def creat_trend(cr_h, t, n=5, hours=72):
    """감지 시점까지 최근 크레아티닌 n개: '-30시간 0.9 → ... → 현재 1.6'"""
    if cr_h is None:
        return "기록 없음"
    ts, vs = cr_h
    m = (ts <= t) & (ts >= t - np.timedelta64(hours, "h"))
    ts, vs = ts[m][-n:], vs[m][-n:]
    parts = []
    for tt, vv in zip(ts, vs):
        dh = (t - tt) / np.timedelta64(1, "h")
        parts.append(("현재 %s" if dh < 0.5 else "-%.0f시간 %%s" % dh) % fmt_num(vv))
    return " → ".join(parts) if parts else "기록 없음"


def patient_line(r):
    sex = "남성" if r["gender"] == "M" else "여성"
    days = (r["t"] - r["admittime"]) / pd.Timedelta(days=1)
    icu_h = (r["t"] - r["intime"]) / H
    return ("[환자] %d세 %s, 입원 %.1f일째, 중환자실 %.0f시간째, 입원 유형: %s, 입원 경로: %s"
            % (int(r["age"]), sex, days, icu_h, r["admission_type"], r["admission_location"]))


def aki_header(r):
    """AKI 가 감지된 시점의 설명"""
    stage = C.kdigo_stage(r["value"], r["base"])
    ratio = r["value"] / r["base"] if r["base"] == r["base"] and r["base"] else None
    rise = r["value"] - r["min48"] if r["min48"] == r["min48"] else None
    detail = ["크레아티닌 %s mg/dL" % fmt_num(r["value"])]
    if ratio:
        detail.append("기저치 %s mg/dL의 %.1f배" % (fmt_num(r["base"]), ratio))
    if rise is not None:
        detail.append("48시간 내 최저값 대비 +%.1f" % rise)
    return ["[감지된 이상] 급성 신손상(AKI, KDIGO 크레아티닌 기준%s): %s"
            % ((" %d단계" % stage) if stage else "", ", ".join(detail))]


def build_prompt(r, header, labs_row, vit_row, aki_txt, other_txt, trend, ask, extra=None):
    lines = [patient_line(r)] + list(header) + ["[크레아티닌 추이] " + trend]
    labs_txt = []
    for key in C.LAB_KEYS:
        v = labs_row.get(key)
        s = fmt_num(v)
        if s is None:
            continue
        _, k, unit, lo, hi, _, _ = C.LAB_INFO[key]
        labs_txt.append("%s %s%s" % (k, s, "↓" if v < lo else ("↑" if v > hi else "")))
    lines.append("[최근 검사 (24시간 이내 최신값)] " + (", ".join(labs_txt) if labs_txt else "기록 없음"))
    vit_txt = []
    for key in C.VITAL_KEYS:
        s = fmt_num(vit_row.get(key))
        if s is not None:
            vit_txt.append("%s %s%s" % (C.VITAL_INFO[key][1], s, C.VITAL_INFO[key][2]))
    lines.append("[최근 활력징후 (6시간 이내)] " + (", ".join(vit_txt) if vit_txt else "기록 없음"))
    lines.append("[투여 중인 AKI 관련 약 (1회 용량×하루 횟수, 분류)] " + (", ".join(aki_txt) if aki_txt else "없음"))
    lines.append("[그 밖의 투여 중인 약] " + (", ".join(other_txt) if other_txt else "없음"))
    lines.extend(extra or [])
    lines.append(ask)
    return "\n".join(lines)


def daily(idx, dose, unit, per24):
    if len(idx) == 0 or not np.all(np.isfinite(dose[idx])) or len(set(unit[idx])) != 1:
        return None
    f = per24[idx]
    has = np.isfinite(f) & (f > 0)
    if has.any() and not has.all():
        return None
    tot = float(np.sum(dose[idx] * (f if has.all() else 1.0)))
    return (tot, unit[idx][0], bool(has.all())) if tot > 0 else None


def aki_events(st, cr):
    """입실 48시간 이후 중환자실에서 감지된 AKI (체류당 최대 2회, 24시간 간격)"""
    e = cr[cr["aki"]].merge(st, on="hadm_id", how="inner")
    e = e[(e["t"] > e["T"]) & (e["t"] <= e["outtime"])]
    e = e[e["t"] + C.WINDOW_HOURS * H <= e["dischtime"]].sort_values(["hadm_id", "t"])
    keep, last = [], {}
    for i, h, t in zip(e.index, e["hadm_id"].values, e["t"].values):
        n, lt = last.get(h, (0, None))
        if n == 0 or (n < C.AKI_MAX_EVENTS and t - lt >= np.timedelta64(C.DEDUP_HOURS, "h")):
            keep.append(i)
            last[h] = (n + 1, t)
    return e.loc[keep].reset_index(drop=True)


def rx_samples(ev, header_fn, ask, cr, labs, vitals, rx, kid=None):
    """ev(hadm_id, t, 환자 정보) 마다 프롬프트와 정답(AKI 관련 약물 처방 변경, t 이후 12시간)을 만든다
    kid: prep_kidney.Kidney 이면 [신장·수분 상태] 줄을 더한다 (사례별 정보 유무는 kid.coverage 에 쌓임)"""
    ev = ev.reset_index(drop=True).copy()
    ev["eid"] = np.arange(len(ev))
    lab_vals = asof_values(ev, labs, range(len(C.LAB_KEYS)), C.LAB_KEYS, C.LAB_LOOKBACK_HOURS)
    vit_vals = asof_values(ev, vitals, range(len(C.VITAL_KEYS)), C.VITAL_KEYS, C.VITAL_LOOKBACK_HOURS)
    cr_by = {h: (g["t"].values, g["value"].values) for h, g in cr[cr["hadm_id"].isin(ev["hadm_id"])].groupby("hadm_id")}
    r = rx[rx["hadm_id"].isin(set(ev["hadm_id"]))].sort_values(["hadm_id", "start"])
    by_h = {h: (g["drug"].values, g["start"].values, g["stop"].values, g["dose"].values.astype("float64"),
                g["unit"].astype(str).values, g["per24"].values.astype("float64"), g["dc_time"].values,
                g["dc_type"].values, g["cat"].values) for h, g in r.groupby("hadm_id")}
    win = np.timedelta64(int(C.WINDOW_HOURS * 3600), "s")
    E = np.array([], "datetime64[ns]")
    empty = (np.array([]), E, E, np.array([]), np.array([]), np.array([]), E, np.array([]), np.array([]))
    prompts, labels, actives = [], [], []
    for _, row in ev.iterrows():
        drugs, stt, spp, dose, unit, per24, dct, dctype, cat = by_h.get(row["hadm_id"], empty)
        t = np.datetime64(row["t"])
        te = t + win
        act = (stt <= t) & (spp > t)
        aki_shown, aki_txt, other = [], [], []
        for i in np.where(act)[0][::-1]:
            d = drugs[i]
            if cat[i] is not None and cat[i] == cat[i]:
                if d not in aki_shown and len(aki_shown) < C.MAX_ACTIVE_MEDS:
                    aki_shown.append(d)
                    aki_txt.append("%s%s [%s]" % (d, fmt_dose(dose[i], unit[i], per24[i]), cat[i]))
            elif d not in other and len(other) < 15:
                other.append(d)
        act_all = set(drugs[act])
        lab = C.empty_label()
        newm = (stt > t) & (stt <= te)
        lab["신규처방"] = [d for d, c in dict.fromkeys(zip(drugs[newm], cat[newm]))
                       if c is not None and c == c and d not in act_all]
        for d in aki_shown:
            is_d = drugs == d
            rows_t = np.where(is_d & act)[0]
            rows_te = np.where(is_d & (stt <= te) & (spp > te))[0]
            restarted = bool(np.any(is_d & newm))
            if len(rows_te) == 0 and not restarted:
                if np.any((dctype[rows_t] == 1) & (dct[rows_t] > t) & (dct[rows_t] <= te)):
                    lab["중단"].append(d)
            elif len(rows_te) > 0:
                if not np.any((dctype[rows_t] > 0) & (dct[rows_t] > t) & (dct[rows_t] <= te)):
                    continue
                d0, d1 = daily(rows_t, dose, unit, per24), daily(rows_te, dose, unit, per24)
                if d0 and d1 and d0[1] == d1[1] and d0[2] == d1[2]:
                    ratio = d1[0] / d0[0]
                    if ratio <= 1 - C.DOSE_CHANGE_RATIO:
                        lab["감량"].append(d)
                    elif ratio >= 1 + C.DOSE_CHANGE_RATIO:
                        lab["증량"].append(d)
        lab_row = lab_vals.loc[row["eid"]].to_dict()
        extra = None
        if kid is not None:
            txt, cov = kid.line(row, lab_row.get("creatinine"))
            extra = [txt]
            kid.coverage.append(cov)
        prompts.append(build_prompt(row, header_fn(row), lab_row,
                                    vit_vals.loc[row["eid"]].to_dict(), aki_txt, other,
                                    creat_trend(cr_by.get(row["hadm_id"]), t), ask, extra))
        labels.append(lab)
        actives.append(aki_shown)
    return ev, prompts, labels, actives


def load_cohort():
    """prep_aki.py·prep_rx.py 공통: _work 의 추출 결과와 성인 첫 중환자실 체류, 크레아티닌 AKI 판정"""
    labs, vitals, rx = load("labs.parquet"), load("vitals.parquet"), load("rx.parquet")
    adm, pat, stays = load("admissions.parquet"), load("patients.parquet"), load("icu_stays.parquet")
    if labs is None or rx is None or adm is None or stays is None:
        raise SystemExit("[오류] 1단계(prep_extract.py)를 먼저 실행하세요.")
    for col, default in (("dose", np.nan), ("unit", ""), ("per24", np.nan), ("dc_time", pd.NaT), ("dc_type", 0)):
        if col not in rx.columns:
            rx[col] = default
    rx["cat"] = rx["drug"].map(C.aki_category)
    st = first_stays(stays, adm, pat)
    cr = creatinine_flags(labs, st)
    return labs, vitals, rx, st, cr


# ---------------------------------------------------------------------------
def main():
    argparse.ArgumentParser().parse_args()
    t0 = time.time()
    labs, vitals, rx, st, cr = load_cohort()
    log("성인 첫 중환자실 체류 %s건" % f"{len(st):,}")
    log("크레아티닌 %s건, AKI 기준 충족 %s건" % (f"{len(cr):,}", f"{int(cr['aki'].sum()):,}"))

    risk, excl = build_risk(st, cr, labs, vitals, rx)
    os.makedirs(C.DATA, exist_ok=True)
    risk.to_parquet(os.path.join(C.DATA, "aki_risk.parquet"), index=False)
    log("위험 예측 표본 %s개 (양성 %.1f%%)" % (f"{len(risk):,}", 100 * risk["label"].mean()))
    ev = aki_events(st, cr)
    log("입실 48시간 이후 AKI 감지 %s건 (처방 모델에 사용)" % f"{len(ev):,}")

    summ = {"definition": {"aki": "KDIGO 크레아티닌: 48시간 내 +%.1f mg/dL 또는 7일 내 최저값의 %.1f배"
                                  % (C.AKI_ABS_RISE, C.AKI_REL_RISE),
                           "risk_design": "이동 창: 입실 %d시간부터 %d시간마다, 직전 %d시간 데이터로 이후 %d시간 안 새 AKI"
                                          % (C.RISK_FIRST_HOURS, C.RISK_STEP_HOURS, C.RISK_WINDOW_HOURS, C.RISK_HORIZON)},
            "first_icu_stays_adult": C.suppress(len(st)),
            "risk": dict(excl, samples={sp: C.suppress((risk["split"] == sp).sum()) for sp in ("train", "val", "test")},
                         positive_rate={sp: round(float(risk.loc[risk["split"] == sp, "label"].mean()), 4)
                                        for sp in ("train", "val", "test") if (risk["split"] == sp).any()},
                         stays={sp: C.suppress(risk.loc[risk["split"] == sp, "stay_key"].nunique())
                                for sp in ("train", "val", "test")},
                         n_features=int(len(risk.columns) - 5)),
            "aki_events_after_48h": {sp: C.suppress((ev["split"] == sp).sum()) for sp in ("train", "val", "test")},
            "kdigo_stage_of_events": {str(k): C.suppress(v) for k, v in Counter(
                C.kdigo_stage(a, b) for a, b in zip(ev["value"], ev["base"])).items()},
            "minutes": round((time.time() - t0) / 60, 1)}
    C.save_json(summ, os.path.join(C.RESULTS, "summary_aki.json"))
    log("AKI 데이터 완료 (%.1f분). 집계: results/summary_aki.json" % summ["minutes"])


if __name__ == "__main__":
    main()
