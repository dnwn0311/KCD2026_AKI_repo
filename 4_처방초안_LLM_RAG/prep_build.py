# -*- coding: utf-8 -*-
"""
2단계: 이상 감지 시점과 그 시점의 환자 상태, 이후 처방 변경(정답)을 묶어 학습 데이터 생성

- 대상: 중환자실 체류 중에 감지된 경우만 (common.ICU_ONLY)
- 감지: common.SCENARIOS 기준. 검사는 결과 등록 시각, 활력징후는 1시간 안 2회 연속 이상
- 같은 입원·같은 상황은 24시간 간격, 최대 3회
- 정답 창(12시간)이 입원 기간 안에 들어오는 경우만 사용
- 정답 (감지 시각 t ~ t+12시간):
    신규처방: 창 안에 시작된 약 중 t 에 투여 중이 아니던 약
    감량/증량: t 와 t+12시간 모두 투여 중이고, 창 안에서 의사가 그 약 오더를 변경(Change)하거나 끊고 다시 냈으며,
               하루 용량(1회 용량 x 하루 횟수)이 10% 이상 줄거나 늘어난 약
    중단: t 에 투여 중이던 약을 창 안에서 의사가 중단(D/C 오더)했고, t+12시간까지 다시 시작하지 않은 약
          (예정된 투여 기간이 끝나서 멈춘 것은 중단으로 보지 않음)
- 입력: 나이·성별·입원 유형/경로·입원·중환자실 경과일, 24시간 내 최신 검사값, 6시간 내 최신 활력징후,
        감지 시점에 투여 중인 약과 용량 (진단 코드는 퇴원 후 확정되므로 쓰지 않음)
- 환자 단위로 train/val/test 분리, 상황·분할별 표본 수 상한

출력: data/{train,val,test}.jsonl (환자 단위, PC 밖으로 내보내지 않음)
      results/summary_build.json (집계만)

사용:  python prep_build.py [--cap-train 3000 --cap-val 300 --cap-test 600]
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import common as C

H = pd.Timedelta(hours=1)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load(name):
    p = os.path.join(C.WORK, name)
    return pd.read_parquet(p) if os.path.exists(p) else None


def detect(labs, vitals):
    """상황별 감지 후보 (hadm_id, subject_id, t, value, scenario)"""
    out = []
    for name, kr, kind, key, op, thr in C.SCENARIOS:
        if kind == "lab":
            if labs is None:
                continue
            code = C.LAB_KEYS.index(key)
            d = labs[labs["code"] == code]
            m = d["value"] < thr if op == "<" else d["value"] > thr
            e = d.loc[m, ["hadm_id", "subject_id", "t", "value"]].copy()
        else:
            if vitals is None:
                continue
            code = C.VITAL_KEYS.index(key)
            d = vitals[vitals["code"] == code].sort_values(["hadm_id", "t"])
            ab = (d["value"] < thr) if op == "<" else (d["value"] > thr)
            prev_ab = ab.groupby(d["hadm_id"]).shift(1, fill_value=False)
            prev_t = d.groupby("hadm_id")["t"].shift(1)
            m = ab & prev_ab & ((d["t"] - prev_t) <= H)
            e = d.loc[m, ["hadm_id", "subject_id", "t", "value"]].copy()
        e["scenario"] = name
        out.append(e)
        log("  %-16s 후보 %s" % (name, f"{len(e):,}"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dedup(ev):
    ev = ev.sort_values(["scenario", "hadm_id", "t"]).reset_index(drop=True)
    keep = np.zeros(len(ev), dtype=bool)
    sc = ev["scenario"].values
    hd = ev["hadm_id"].values
    tt = ev["t"].values.astype("datetime64[ns]").astype("int64")
    gap = int(C.DEDUP_HOURS * 3600 * 1e9)
    last_key, last_t, cnt = None, None, 0
    for i in range(len(ev)):
        k = (sc[i], hd[i])
        if k != last_key:
            last_key, last_t, cnt = k, tt[i], 1
            keep[i] = True
        elif cnt < C.MAX_PER_ADMISSION and tt[i] - last_t >= gap:
            last_t, cnt = tt[i], cnt + 1
            keep[i] = True
    return ev[keep].reset_index(drop=True)


def asof_values(events, table, codes, keys, lookback_h):
    """각 감지 시점에서 항목별 lookback 시간 이내 최신값"""
    res = {}
    ev = events[["eid", "hadm_id", "t"]].sort_values("t")
    tol = pd.Timedelta(hours=lookback_h)
    for code, key in zip(codes, keys):
        if table is None:
            res[key] = pd.Series(np.nan, index=events["eid"])
            continue
        d = table.loc[table["code"] == code, ["hadm_id", "t", "value"]].sort_values("t")
        m = pd.merge_asof(ev, d, on="t", by="hadm_id", direction="backward", tolerance=tol,
                          allow_exact_matches=True)
        res[key] = m.set_index("eid")["value"]
    return pd.DataFrame(res).reindex(events["eid"])


def fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    return ("%.1f" % v) if abs(v) < 100 else ("%.0f" % v)


def fmt_dose(dose, unit, per24):
    if dose is None or not np.isfinite(dose):
        return ""
    txt = "%g%s" % (round(float(dose), 3), unit or "")
    if per24 is not None and np.isfinite(per24) and per24 > 0:
        txt += "×%g/일" % round(float(per24), 2)
    return " " + txt


def build_prompt(ev, labs_row, vit_row, active):
    age = int(ev["age"])
    sex = "남성" if ev["gender"] == "M" else "여성"
    days = (ev["t"] - ev["admittime"]) / pd.Timedelta(days=1)
    name, kr = ev["scenario"], C.SCENARIO_KR[ev["scenario"]]
    sc = [s for s in C.SCENARIOS if s[0] == name][0]
    if sc[2] == "lab":
        info = C.LAB_INFO[sc[3]]
        trig = "%s %s %s" % (info[1], fmt_num(ev["value"]), info[2])
    else:
        info = C.VITAL_INFO[sc[3]]
        trig = "%s %s %s (1시간 안 2회 연속)" % (info[1], fmt_num(ev["value"]), info[2])
    lines = [
        "[환자] %d세 %s, 입원 %.1f일째, 중환자실 %.1f일째, 입원 유형: %s, 입원 경로: %s"
        % (age, sex, days, ev["icu_days"], ev["admission_type"], ev["admission_location"]),
        "[감지된 이상] %s: %s" % (kr, trig),
    ]
    labs_txt = []
    for key in C.LAB_KEYS:
        v = labs_row.get(key)
        s = fmt_num(v)
        if s is None:
            continue
        _, k, unit, lo, hi, _, _ = C.LAB_INFO[key]
        mark = "↓" if v < lo else ("↑" if v > hi else "")
        labs_txt.append("%s %s%s" % (k, s, mark))
    lines.append("[최근 검사 (24시간 이내 최신값)] " + (", ".join(labs_txt) if labs_txt else "기록 없음"))
    vit_txt = []
    for key in C.VITAL_KEYS:
        s = fmt_num(vit_row.get(key))
        if s is not None:
            vit_txt.append("%s %s%s" % (C.VITAL_INFO[key][1], s, C.VITAL_INFO[key][2]))
    lines.append("[최근 활력징후 (6시간 이내)] " + (", ".join(vit_txt) if vit_txt else "중환자실 기록 없음"))
    lines.append("[현재 투여 중인 약 (1회 용량×하루 횟수)] " + (", ".join(active) if active else "없음"))
    lines.append("이 환자에게 필요한 처방 변경(신규 시작, 감량, 증량, 중단) 초안을 제안하세요.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-train", type=int, default=3000)
    ap.add_argument("--cap-val", type=int, default=300)
    ap.add_argument("--cap-test", type=int, default=600)
    ap.add_argument("--min-drug-count", type=int, default=20, help="train 정답에서 이 횟수 미만인 약은 제외")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    caps = {"train": a.cap_train, "val": a.cap_val, "test": a.cap_test}
    os.makedirs(C.DATA, exist_ok=True)

    labs, vitals, rx = load("labs.parquet"), load("vitals.parquet"), load("rx.parquet")
    adm, pat = load("admissions.parquet"), load("patients.parquet")
    if labs is None or rx is None or adm is None:
        raise SystemExit("[오류] 1단계(prep_extract.py)를 먼저 실행하세요.")
    log("감지 후보 계산")
    ev = detect(labs, vitals)
    n_raw = ev["scenario"].value_counts().to_dict()
    ev = dedup(ev)
    n_dedup = ev["scenario"].value_counts().to_dict()

    adm = adm.copy()
    adm[["admission_type", "admission_location"]] = adm[["admission_type", "admission_location"]].fillna("미상")
    ev = ev.merge(adm[["hadm_id", "admittime", "dischtime", "admission_type", "admission_location"]],
                  on="hadm_id", how="inner")
    ev = ev[ev["t"] + pd.Timedelta(hours=C.WINDOW_HOURS) <= ev["dischtime"]]
    n_window = ev["scenario"].value_counts().to_dict()

    # 중환자실 체류 중 감지만
    stays = load("icu_stays.parquet")
    if C.ICU_ONLY:
        if stays is None:
            raise SystemExit("[오류] 중환자실 체류 기간(_work/icu_stays.parquet)이 없습니다. prep_extract.py 를 다시 실행하세요.")
        ev = ev.reset_index(drop=True)
        ev["_i"] = np.arange(len(ev))
        m = ev[["_i", "hadm_id", "t"]].merge(stays[["hadm_id", "intime", "outtime"]], on="hadm_id", how="inner")
        m = m[(m["t"] >= m["intime"]) & (m["t"] <= m["outtime"])].sort_values("intime").drop_duplicates("_i")
        ev = ev.merge(m[["_i", "intime"]], on="_i", how="inner").drop(columns=["_i"])
        ev["icu_days"] = (ev["t"] - ev["intime"]) / pd.Timedelta(days=1)
    else:
        ev["icu_days"] = np.nan
    n_icu = ev["scenario"].value_counts().to_dict()
    ev = ev.merge(pat, on="subject_id", how="inner")
    ev["age"] = ev["anchor_age"] + (ev["t"].dt.year - ev["anchor_year"])
    ev["split"] = ev["subject_id"].map(C.subject_split)

    # 상황·분할별 표본 수 상한
    rng = np.random.RandomState(a.seed)
    picked = []
    for (s, sp), g in ev.groupby(["scenario", "split"]):
        n = min(len(g), caps[sp])
        picked.append(g.iloc[rng.permutation(len(g))[:n]])
    ev = pd.concat(picked, ignore_index=True)
    ev["eid"] = np.arange(len(ev))
    log("표본 %s건으로 상태 요약 생성" % f"{len(ev):,}")

    lab_vals = asof_values(ev, labs, range(len(C.LAB_KEYS)), C.LAB_KEYS, C.LAB_LOOKBACK_HOURS)
    vit_vals = asof_values(ev, vitals, range(len(C.VITAL_KEYS)), C.VITAL_KEYS, C.VITAL_LOOKBACK_HOURS)

    # 처방: 입원별로 묶어 두고 시점마다 투여 중 / 신규 / 감량·증량 / 중단 계산
    rx = rx[rx["hadm_id"].isin(set(ev["hadm_id"]))].sort_values(["hadm_id", "start"])
    for col, default in (("dose", np.nan), ("unit", ""), ("per24", np.nan), ("dc_time", pd.NaT), ("dc_type", 0)):
        if col not in rx.columns:
            rx[col] = default
    rx["unit"] = rx["unit"].astype(str)
    by_h = {h: (g["drug"].values, g["start"].values, g["stop"].values, g["dose"].values.astype("float64"),
                g["unit"].values, g["per24"].values.astype("float64"), g["dc_time"].values, g["dc_type"].values)
            for h, g in rx.groupby("hadm_id")}
    win = np.timedelta64(int(C.WINDOW_HOURS * 3600), "s")
    empty = (np.array([]), np.array([], "datetime64[ns]"), np.array([], "datetime64[ns]"), np.array([]),
             np.array([]), np.array([]), np.array([], "datetime64[ns]"), np.array([]))

    def daily(idx, dose, unit, per24):
        """하루 용량 (같은 단위일 때만). 계산할 수 없으면 None"""
        if len(idx) == 0 or not np.all(np.isfinite(dose[idx])) or len(set(unit[idx])) != 1:
            return None
        f = per24[idx]
        has = np.isfinite(f) & (f > 0)
        if has.any() and not has.all():
            return None
        tot = float(np.sum(dose[idx] * (f if has.all() else 1.0)))
        return (tot, unit[idx][0], bool(has.all())) if tot > 0 else None

    actives, active_names, labels = [], [], []
    for _, r in ev.iterrows():
        drugs, st, sp, dose, unit, per24, dct, dctype = by_h.get(r["hadm_id"], empty)
        t = np.datetime64(r["t"])
        te = t + win
        act_mask = (st <= t) & (sp > t)
        act_idx = np.where(act_mask)[0][::-1]  # 최근 시작 순
        shown, txt = [], []
        for i in act_idx:
            d = drugs[i]
            if d in shown:
                continue
            shown.append(d)
            txt.append(d + fmt_dose(dose[i], unit[i], per24[i]))
            if len(shown) >= C.MAX_ACTIVE_MEDS:
                break
        act_all = set(drugs[act_mask])
        lab = C.empty_label()
        new_mask = (st > t) & (st <= te)
        lab["신규처방"] = [d for d in dict.fromkeys(drugs[new_mask]) if d not in act_all]
        for d in shown:  # 감량·증량·중단은 입력에 보여 준 약 중에서만
            is_d = drugs == d
            rows_t = np.where(is_d & act_mask)[0]
            rows_te = np.where(is_d & (st <= te) & (sp > te))[0]
            restarted = bool(np.any(is_d & new_mask))
            if len(rows_te) == 0 and not restarted:
                dcm = (dctype[rows_t] == 1) & (dct[rows_t] > t) & (dct[rows_t] <= te)
                if np.any(dcm):
                    lab["중단"].append(d)
            elif len(rows_te) > 0:
                # 의사가 창 안에서 그 약의 오더를 바꾸거나(Change) 끊고 다시 낸(D/C) 경우만 용량 변경으로 봄
                # (같은 약의 다른 오더가 예정대로 끝나 합계만 달라진 경우를 제외)
                acted = np.any((dctype[rows_t] > 0) & (dct[rows_t] > t) & (dct[rows_t] <= te))
                if not acted:
                    continue
                d0, d1 = daily(rows_t, dose, unit, per24), daily(rows_te, dose, unit, per24)
                if d0 and d1 and d0[1] == d1[1] and d0[2] == d1[2]:
                    ratio = d1[0] / d0[0]
                    if ratio <= 1 - C.DOSE_CHANGE_RATIO:
                        lab["감량"].append(d)
                    elif ratio >= 1 + C.DOSE_CHANGE_RATIO:
                        lab["증량"].append(d)
        actives.append(txt)
        active_names.append(shown)
        labels.append(lab)
    ev["active"] = actives
    ev["active_names"] = active_names
    ev["label_raw"] = labels

    # 정답 약물 사전: train 정답(모든 변경 종류)에서 충분히 나온 약만
    cnt = Counter(d for L, sp in zip(ev["label_raw"], ev["split"]) if sp == "train" for k in C.ACTIONS for d in L[k])
    vocab = {d for d, n in cnt.items() if n >= a.min_drug_count}
    ev["label"] = [{k: [d for d in L[k] if d in vocab][:C.MAX_LABEL_MEDS] for k in C.ACTIONS}
                   for L in ev["label_raw"]]

    files = {sp: open(os.path.join(C.DATA, "%s.jsonl" % sp), "w", encoding="utf-8") for sp in caps}
    lens = defaultdict(list)
    for i, r in ev.iterrows():
        prompt = build_prompt(r, lab_vals.loc[r["eid"]].to_dict(), vit_vals.loc[r["eid"]].to_dict(), r["active"])
        ans = {"상황": C.SCENARIO_KR[r["scenario"]]}
        ans.update(r["label"])
        answer = json.dumps(ans, ensure_ascii=False)
        rec = {"scenario": r["scenario"], "label": r["label"], "active": r["active_names"],
               "messages": [{"role": "system", "content": C.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]}
        files[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        lens[r["split"]].append(len(prompt))
    for f in files.values():
        f.close()

    # 집계 요약 (10 미만은 <10)
    summ = {"window_hours": C.WINDOW_HOURS, "vocab_size": len(vocab),
            "candidates": {k: C.suppress(v) for k, v in n_raw.items()},
            "after_dedup": {k: C.suppress(v) for k, v in n_dedup.items()},
            "after_window": {k: C.suppress(v) for k, v in n_window.items()},
            "in_icu": {k: C.suppress(v) for k, v in n_icu.items()},
            "samples": {}, "label_empty_rate": {}, "action_rate": {}, "mean_label_size": {}, "top_drugs_train": {},
            "vitals_available_rate": {}, "prompt_chars_mean": {sp: int(np.mean(v)) if v else 0
                                                              for sp, v in lens.items()}}
    for s in C.SCENARIO_KR:
        g = ev[ev["scenario"] == s]
        if len(g) == 0:
            continue
        summ["samples"][s] = {sp: C.suppress((g["split"] == sp).sum()) for sp in caps}
        summ["label_empty_rate"][s] = round(float(np.mean([sum(len(v) for v in L.values()) == 0
                                                           for L in g["label"]])), 3)
        summ["action_rate"][s] = {k: round(float(np.mean([len(L[k]) > 0 for L in g["label"]])), 3)
                                  for k in C.ACTIONS}
        summ["mean_label_size"][s] = {k: round(float(np.mean([len(L[k]) for L in g["label"]])), 2)
                                      for k in C.ACTIONS}
        summ["top_drugs_train"][s] = {}
        for k in C.ACTIONS:
            c = Counter(d for L in g.loc[g["split"] == "train", "label"] for d in L[k])
            summ["top_drugs_train"][s][k] = [[d, C.suppress(n)] for d, n in c.most_common(8)]
        vv = vit_vals.loc[g["eid"]]
        summ["vitals_available_rate"][s] = round(float(vv.notna().any(axis=1).mean()), 3)
    summ["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(summ, os.path.join(C.RESULTS, "summary_build.json"))
    log("2단계 완료 (%.1f분). 집계: results/summary_build.json" % summ["minutes"])


if __name__ == "__main__":
    main()
