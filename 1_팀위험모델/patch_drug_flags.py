"""NSAID·ACEi/ARB 약물 플래그를 성분명 기준으로 바로잡은 팀 폴더(_work/team_fixed)를 만든다

왜 필요한가
  팀원 전처리(06단계)는 성분마다 RxClass 에서 ATC 코드를 받아 오는데, 이 값에는 그 성분이 들어간 복합제 분류도 같이 온다.
  예) diphenhydramine -> D04AA;M01AE;R06AA  (M01AE 는 NSAID 계열) -> diphenhydramine 만 처방받아도 NSAID 플래그가 1
      amlodipine      -> C08CA;C09BB;C09DB;C09DX;C09XA;C10BX (C09 = ACEi/ARB 계열) -> amlodipine 만 처방받아도 ACEi/ARB 플래그가 1
  그래서 atc_M01A_nsaid 와 atc_C09_ace_arb 는 실제보다 부풀려져 있다.

하는 일 (원본은 건드리지 않는다)
  1) 팀원 전처리와 같은 규칙(예측 시점 전에 시작, ICU 입실 전에 끝나지 않은 처방)으로 처방을 모으고
  2) NSAID / ACEi·ARB 를 성분명 목록으로 다시 정의해 플래그를 만들고 (피부·눈·귀 등 국소 경로 제외)
  3) 이전 값과 비교한 표를 보여 주고
  4) wide.parquet 의 해당 열과 n_kidney_relevant_classes 를 바꾼 복사본을 _work/team_fixed/ 에 쓴다
     (그 밖의 파일은 그대로 복사)
  atc3_M01A 도 NSAID 와 같은 정의이므로 같이 바꾼다. 다른 ATC 3단계 플래그는 그대로 둔다.

사용 (WSL, duckdb 가 있는 aki_venv 로):
  cd "/mnt/c/Users/User/Documents/KCD_처방추천"
  ~/aki_venv/bin/python patch_drug_flags.py
그다음 같은 분할로 모델을 다시 학습해 비교 (kcd_venv):
  python train_risk_trend.py --team-dir _work/team_fixed
  -> 기존 결과 (기준 0.7482 / +추세 0.7687) 와 비교
"""
import argparse
import os
import shutil
import sys

import duckdb

NSAID = ["ibuprofen", "ketorolac", "naproxen", "diclofenac", "indomethacin", "celecoxib", "meloxicam", "ketoprofen",
         "flurbiprofen", "etodolac", "nabumetone", "piroxicam", "sulindac", "oxaprozin", "mefenamic acid", "diflunisal",
         "salsalate", "tolmetin", "fenoprofen"]
ACE_ARB = ["lisinopril", "enalapril", "enalaprilat", "captopril", "ramipril", "benazepril", "quinapril", "fosinopril",
           "perindopril", "trandolapril", "moexipril", "losartan", "valsartan", "irbesartan", "candesartan", "olmesartan",
           "telmisartan", "azilsartan", "eprosartan"]
LOCAL = "EYE|EAR|^TP$|^OU$|^OD$|^OS$|INTRAVITREAL|^IVT$|TOPICAL|OPHTH|OTIC|NASAL|INH"   # 국소·비전신 경로
KIDNEY_FLAGS = ["atc_J01G_aminoglycoside", "atc_J01XA_vancomycin", "drug_pip_tazo", "atc_J01XB_polymyxin",
                "atc_J02AA_amphotericin", "atc_J05AB_nucleoside_antiviral", "atc_M01A_nsaid", "atc_C03C_loop_diuretic",
                "atc_C09_ace_arb", "atc_L04AD_calcineurin"]
COPY = ["cohort_final.parquet", "splits.parquet", "default_features.json", "trend_features.parquet"]


def flag(con, ings):
    lst = ",".join("'%s'" % i for i in ings)
    return {r[0] for r in con.execute(
        "select distinct stay_id from rx_i where ing in (%s) and not regexp_matches(coalesce(route, ''), '%s')" % (lst, LOCAL)).fetchall()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=os.path.expanduser("~/aki_pipeline"))
    ap.add_argument("--team-dir", default="_work/team")
    ap.add_argument("--out", default="_work/team_fixed")
    ap.add_argument("--inplace", action="store_true",
                    help="team-dir 의 wide.parquet 을 직접 고친다 (원본은 wide_before_fix.parquet 로 한 번 백업, 다시 실행해도 원본에서 시작). 이동 시점 폴더용")
    a = ap.parse_args()
    if a.inplace:
        a.out = a.team_dir
        orig, bak = os.path.join(a.team_dir, "wide.parquet"), os.path.join(a.team_dir, "wide_before_fix.parquet")
        if not os.path.exists(bak):
            if not os.path.exists(orig):
                sys.exit("[오류] %s 가 없습니다" % orig)
            shutil.copy2(orig, bak)
    P = os.path.join(a.proj, "data/mimiciv/3.1_parquet/hosp/prescriptions.parquet")
    DM = os.path.join(a.proj, "data/external/drug_map.parquet")
    W = os.path.join(a.team_dir, "wide_before_fix.parquet" if a.inplace else "wide.parquet")
    for p in (P, DM, W):
        if not os.path.exists(p):
            sys.exit("[오류] %s 가 없습니다" % p)

    con = duckdb.connect()
    con.execute("create temp table c as select stay_id, hadm_id, intime, time_zero from '%s'" % W)
    con.execute("""
create temp table o as
select c.stay_id, lower(p.drug) drug, upper(p.route) route,
  case when p.ndc is null or p.ndc in ('0', '') then null else p.ndc end ndc
from c join '%s' p on p.hadm_id = c.hadm_id
where p.starttime < c.time_zero and coalesce(p.stoptime, p.starttime) >= c.intime""" % P)
    con.execute("""
create temp table rx_i as
select distinct stay_id, route, trim(ing) ing from (
  select o.stay_id, o.route, unnest(string_split(lower(coalesce(nullif(mn.ingredients, ''), nullif(mm.ingredients, ''), '')), ';')) ing
  from o left join '%s' mn on mn.kind = 'ndc' and mn.key = o.ndc
         left join '%s' mm on mm.kind = 'name' and mm.key = o.drug) where ing <> ''""" % (DM, DM))

    new = {"atc_M01A_nsaid": flag(con, NSAID), "atc_C09_ace_arb": flag(con, ACE_ARB)}
    new["atc3_M01A"] = new["atc_M01A_nsaid"]
    w = con.execute("select * from '%s'" % W).df()
    n = len(w)
    print("코호트 %d건" % n)
    print("플래그               이전(ATC)  수정(성분명)  겹침   이전에만  수정에만")
    for col in ("atc_M01A_nsaid", "atc_C09_ace_arb", "atc3_M01A"):
        if col not in w:
            print("  %-20s (wide 에 없음)" % col); continue
        old = set(w.loc[w[col].fillna(False).astype(bool), "stay_id"])
        nw = new[col]
        print("  %-20s %7d   %9d  %6d  %7d  %7d" % (col, len(old), len(nw), len(old & nw), len(old - nw), len(nw - old)))
        w[col] = w["stay_id"].isin(nw)
    w["n_kidney_relevant_classes"] = sum(w[c].fillna(False).astype(bool).astype(int) for c in KIDNEY_FLAGS if c in w)

    os.makedirs(a.out, exist_ok=True)
    for f in ([] if a.inplace else COPY):
        s = os.path.join(a.team_dir, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.out, f))
    con.register("w_fixed", w)
    con.execute("copy (select * from w_fixed) to '%s' (format parquet)" % os.path.join(a.out, "wide.parquet").replace("'", "''"))
    print("저장: %s (wide.parquet 은 수정본, 나머지는 복사)" % a.out)
    print("다음:  python train_risk_trend.py --team-dir %s" % a.out)


if __name__ == "__main__":
    main()
