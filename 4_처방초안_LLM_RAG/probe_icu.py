# -*- coding: utf-8 -*-
"""
추가 데이터 점검: 소변량·투석·체중·섭취량을 쓸 수 있는지 확인한다 (집계만 출력, 환자 단위 자료는 출력하지 않음)

확인하는 것
  1) outputevents / procedureevents / d_items / inputevents / icustays 파일이 MIMIC_ROOT 아래에 있는지
  2) d_items 에서 소변량·투석·체중 관련 항목 번호와 이름 (항목 사전이라 환자 정보 없음)
  3) d_labitems 에 시스타틴 C, 요산, 요중 크레아티닌이 있는지
  4) 소변량(outputevents)·투석(procedureevents) 항목별 기록 수, 체중(inputevents.patientweight) 채워진 비율

사용:  python probe_icu.py        결과: results/probe_icu.json
"""
import json
import os
import re
import time

import pandas as pd

import common as C

ITEM_PAT = re.compile(r"urine|foley|void|nephrostomy|ureter|suprapubic|cath|dialy|crrt|cvvh|ultrafiltr|blood flow|"
                      r"filtrat|weight|hemofilt|pre.?filter|replacement fluid|effluent", re.I)
LAB_PAT = re.compile(r"cystatin|uric acid|creatinine, urine|egfr|gfr|urea nitrogen", re.I)
CHUNK = 2_000_000


def find(prefix, max_depth=4):
    base = C.MIMIC.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(C.MIMIC):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        for f in sorted(files):
            if f.startswith(prefix) and f.endswith((".csv", ".csv.gz")) and ".part" not in f:
                return os.path.join(d, f)
    return None


def sup(n):
    return C.suppress(n)


def read_chunks(path, cols, limit_rows=None):
    got = 0
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        yield ch
        got += len(ch)
        if limit_rows and got >= limit_rows:
            break


def main():
    t0 = time.time()
    out = {"mimic_root": C.MIMIC, "files": {}}
    paths = {k: find(k) for k in ("outputevents", "procedureevents", "d_items", "inputevents", "icustays")}
    for k, p in paths.items():
        out["files"][k] = {"found": bool(p), "name": os.path.basename(p) if p else None,
                           "gb": round(os.path.getsize(p) / 1e9, 2) if p else None}
        print("[파일] %-16s %s" % (k, "있음 (%s)" % os.path.basename(p) if p else "없음"), flush=True)

    labels = {}
    if paths["d_items"]:
        di = pd.read_csv(paths["d_items"])
        hit = di[di["label"].astype(str).str.contains(ITEM_PAT)]
        labels = dict(zip(di["itemid"], di["label"]))
        out["d_items_related"] = [{"itemid": int(r.itemid), "label": r.label, "linksto": r.linksto,
                                   "category": r.category} for r in hit.itertuples()][:300]
        print("[d_items] 관련 항목 %d개" % len(hit), flush=True)
    else:
        out["d_items_related"] = None
        print("[d_items] 없음 - 투석 항목 번호를 이름으로 찾을 수 없습니다. PhysioNet 의 icu/d_items.csv.gz 가 필요합니다")

    dl = find("d_labitems")
    if dl:
        d = pd.read_csv(dl)
        hit = d[d["label"].astype(str).str.contains(LAB_PAT)]
        out["d_labitems_related"] = [{"itemid": int(r.itemid), "label": r.label, "fluid": r.fluid} for r in hit.itertuples()]
        print("[d_labitems] 시스타틴 C: %s" % ("있음" if hit["label"].str.contains("cystatin", case=False).any() else "없음"))

    if paths["outputevents"]:
        cnt = {}
        for ch in read_chunks(paths["outputevents"], ["itemid"]):
            for k, v in ch["itemid"].value_counts().items():
                cnt[int(k)] = cnt.get(int(k), 0) + int(v)
        top = sorted(cnt.items(), key=lambda x: -x[1])[:25]
        out["outputevents_top_items"] = [{"itemid": k, "label": labels.get(k), "rows": sup(v)} for k, v in top]
        print("[outputevents] 항목 %d종, 상위: %s" % (len(cnt), [(k, labels.get(k), v) for k, v in top[:8]]), flush=True)

    if paths["procedureevents"]:
        cnt = {}
        for ch in read_chunks(paths["procedureevents"], ["itemid"]):
            for k, v in ch["itemid"].value_counts().items():
                cnt[int(k)] = cnt.get(int(k), 0) + int(v)
        rel = {k: v for k, v in cnt.items() if labels and ITEM_PAT.search(str(labels.get(k, "")))}
        out["procedureevents_dialysis_items"] = [{"itemid": k, "label": labels.get(k), "rows": sup(v)}
                                                 for k, v in sorted(rel.items(), key=lambda x: -x[1])]
        print("[procedureevents] 항목 %d종, 투석 관련 %d종" % (len(cnt), len(rel)), flush=True)

    if paths["inputevents"]:
        n = nn = 0
        w = []
        for ch in read_chunks(paths["inputevents"], ["patientweight"], limit_rows=3_000_000):
            n += len(ch)
            nn += int(ch["patientweight"].notna().sum())
            w.append(ch["patientweight"].dropna().sample(min(20000, int(ch["patientweight"].notna().sum()))
                                                          if ch["patientweight"].notna().any() else 0, random_state=1))
        w = pd.concat(w) if w else pd.Series(dtype=float)
        out["inputevents_patientweight"] = {"rows_checked": n, "filled_rate": round(nn / n, 4) if n else None,
                                            "quantiles_kg": {str(q): round(float(w.quantile(q)), 1) for q in (0.01, 0.5, 0.99)}
                                            if len(w) else None}
        print("[inputevents] 체중 채워진 비율 %s (확인한 %s행)" % (out["inputevents_patientweight"]["filled_rate"], f"{n:,}"))

    out["minutes"] = round((time.time() - t0) / 60, 1)
    C.save_json(out, os.path.join(C.RESULTS, "probe_icu.json"))
    print("[완료] results/probe_icu.json (%.1f분)" % out["minutes"])


if __name__ == "__main__":
    main()
