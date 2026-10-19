# -*- coding: utf-8 -*-
"""
팀원 파이프라인(01단계)이 읽는 자리에 MIMIC-IV 파일을 놓는다.

팀원 코드는 data/mimiciv/3.1/{hosp,icu}/<표 이름>.csv (압축 안 한 CSV) 를 읽는다.
D:\\MIMIC 은 파일 이름·위치·압축 여부가 제각각이므로 아래처럼 맞춘다.
  - 압축 안 한 CSV (예: chartevents-001.csv, labevents-005.csv) -> 바로 가기(심볼릭 링크). 복사하지 않음
  - .csv.gz -> 목적지에 풀어서 저장 (작은 표만 해당)
각 후보 파일은 첫 줄(열 이름)을 읽어 그 표가 맞는지 확인한다. 이름이 비슷해도 열이 다르면 쓰지 않는다.
환자 단위 내용은 출력하지 않는다 (파일 이름·크기·열 이름만).

사용:  python team_link_mimic.py --mimic /mnt/d/MIMIC --dest ~/aki_pipeline/data/mimiciv/3.1
"""
import argparse
import gzip
import os
import re
import shutil
import sys

# 표 -> (폴더, 첫 줄에 있어야 하는 열)
TABLES = {
    "patients": ("hosp", {"subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"}),
    "admissions": ("hosp", {"subject_id", "hadm_id", "admittime", "dischtime", "admission_type"}),
    "labevents": ("hosp", {"labevent_id", "subject_id", "itemid", "charttime", "valuenum"}),
    "d_labitems": ("hosp", {"itemid", "label", "fluid", "category"}),
    "diagnoses_icd": ("hosp", {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"}),
    "d_icd_diagnoses": ("hosp", {"icd_code", "icd_version", "long_title"}),
    "prescriptions": ("hosp", {"subject_id", "hadm_id", "pharmacy_id", "drug_type", "drug"}),
    "icustays": ("icu", {"subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"}),
    "chartevents": ("icu", {"subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"}),
    "d_items": ("icu", {"itemid", "label", "linksto", "category"}),
    "inputevents": ("icu", {"subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"}),
    "outputevents": ("icu", {"subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"}),
    "procedureevents": ("icu", {"subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"}),
}
BAD_MARK = (".part", ".crdownload", ".tmp", ".plan")


def header_of(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        return set(f.readline().strip().replace('"', "").split(","))


def candidates(root, table, max_depth=4):
    base = root.rstrip(os.sep).count(os.sep)
    out = []
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        for f in files:
            low = f.lower()
            if any(m in low for m in BAD_MARK) or not low.startswith(table):
                continue
            rest = low[len(table):]
            if not (rest.startswith(("-", "_", ".")) and (low.endswith(".csv") or low.endswith(".csv.gz"))):
                continue
            # 'prescriptions' 가 'prescriptions_extra' 같은 다른 표와 섞이지 않게: 숫자 꼬리표만 허용
            if not re.fullmatch(r"([-_.]\d+)*\.csv(\.gz)?", rest):
                continue
            out.append(os.path.join(d, f))
    # 압축 안 한 CSV 를 먼저 (바로 가기로 쓸 수 있고 빠름), 그다음 gz, 큰 파일 먼저
    out.sort(key=lambda p: (p.endswith(".gz"), -os.path.getsize(p)))
    return out


def decompress(src, dst):
    tmp = dst + ".tmp"
    with gzip.open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fails = []
    for table, (folder, need) in TABLES.items():
        out_dir = os.path.join(a.dest, folder)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, table + ".csv")
        if os.path.lexists(dst) and os.path.getsize(dst) > 0:
            print("[있음] %-16s %s" % (table, dst))
            continue
        if os.path.islink(dst):
            os.remove(dst)
        chosen, notes = None, []
        for p in candidates(a.mimic, table):
            try:
                hdr = header_of(p)
            except Exception as e:  # 깨진 gz 등
                notes.append("%s: 읽을 수 없음 (%s)" % (os.path.basename(p), type(e).__name__))
                continue
            if need <= hdr:
                chosen = p
                break
            notes.append("%s: 열이 맞지 않음" % os.path.basename(p))
        if not chosen:
            fails.append(table)
            print("[없음] %-16s 쓸 수 있는 파일을 찾지 못했습니다. %s" % (table, "; ".join(notes) or "후보 없음"))
            continue
        gb = os.path.getsize(chosen) / 1e9
        if a.dry_run:
            print("[후보] %-16s %s (%.2f GB)" % (table, chosen, gb))
            continue
        if chosen.endswith(".gz"):
            print("[풀기] %-16s %s (%.2f GB 압축) ..." % (table, os.path.basename(chosen), gb), flush=True)
            try:
                decompress(chosen, dst)
            except Exception as e:
                fails.append(table)
                print("  실패: %s: %s -> 파일이 손상됐을 수 있습니다. 다시 받아 주세요" % (type(e).__name__, e))
                continue
            print("  완료 %.2f GB" % (os.path.getsize(dst) / 1e9))
        else:
            os.symlink(os.path.abspath(chosen), dst)
            print("[연결] %-16s %s (%.2f GB)" % (table, os.path.basename(chosen), gb))
    if fails:
        print("\n[오류] 다음 표를 준비하지 못했습니다: %s" % ", ".join(fails))
        sys.exit(2)
    print("\n[완료] 13개 표 준비됨: %s" % a.dest)


if __name__ == "__main__":
    main()
