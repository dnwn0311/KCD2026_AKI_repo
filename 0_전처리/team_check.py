# -*- coding: utf-8 -*-
"""
팀원 파이프라인 실행 결과가 팀원이 보고한 숫자와 같은지 확인한다 (집계 숫자만 출력).

  python team_check.py --stage tables  --proj ~/aki_pipeline     # 01단계 뒤: 변환된 표의 행 수
  python team_check.py --stage cohort  --proj ~/aki_pipeline     # 05단계 뒤: 코호트 규칙별 남은 입실 수
  python team_check.py --stage final   --proj ~/aki_pipeline     # 11단계 뒤: wide 표 크기·분할·양성 수

기대값은 팀원 PREPROCESSING_REPORT.md(MIMIC-IV 3.1)에서 가져왔다. 같으면 내 PC의 MIMIC 파일이 완전하고
버전이 같다는 뜻이다. 다르면 파일이 덜 받아졌거나 버전이 다른 것이므로 이후 단계로 넘어가지 말고 확인한다.
종료 코드: 모두 일치 0, 하나라도 다르면 3.
"""
import argparse
import os
import sys

import duckdb

# 팀원 보고서의 값
COHORT_EXPECTED = {"stays": 11000, "patients": 10211, "positives": 1733}
FLOW_EXPECTED = [  # (규칙, 남은 입실 수)  - flow_base.csv / flow_final.csv 의 마지막 값과 비교
    ("전체 ICU 입실", 94458),
    ("최종 코호트", 11000),
]
SPLIT_EXPECTED = {"split_random": {"train": 7689, "val": 1653, "test": 1658}}
WIDE_EXPECTED_ROWS, WIDE_EXPECTED_COLS = 11000, 350
# MIMIC-IV 3.1 표 크기(대략). 5% 이상 다르면 알린다. 입실 수(icustays)는 팀원 보고서에서 확인된 값
TABLE_EXPECTED = {
    "hosp/patients": 364627, "hosp/admissions": 546028, "hosp/labevents": 158374764,
    "hosp/prescriptions": 20292611, "hosp/diagnoses_icd": 6364488,
    "icu/icustays": 94458, "icu/chartevents": 432997491, "icu/inputevents": 10953713,
    "icu/outputevents": 5359395, "icu/procedureevents": 808706,
}


def show(name, got, exp, tol=0.0):
    ok = abs(got - exp) <= tol * exp if exp else got == exp
    print("  [%s] %-34s 내 PC %s | 팀원 보고서 %s" % ("일치" if ok else "다름", name, f"{got:,}", f"{exp:,}"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tables", "cohort", "final"])
    ap.add_argument("--proj", default=os.path.expanduser("~/aki_pipeline"))
    a = ap.parse_args()
    os.chdir(a.proj)
    con = duckdb.connect()
    oks = []
    if a.stage == "tables":
        print("[01단계 결과] 변환된 표의 행 수 (대략값과 5% 이내면 정상)")
        for t, exp in TABLE_EXPECTED.items():
            p = "data/mimiciv/3.1_parquet/%s.parquet" % t
            if not os.path.exists(p):
                print("  [없음] %s" % p)
                oks.append(False)
                continue
            n = con.execute("select count(*) from '%s'" % p).fetchone()[0]
            oks.append(show(t, n, exp, tol=0.05))
        # ICU 입실 수는 정확히 같아야 한다
        n = con.execute("select count(*) from 'data/mimiciv/3.1_parquet/icu/icustays.parquet'").fetchone()[0]
        oks.append(show("icu/icustays (정확히)", n, 94458))
    elif a.stage == "cohort":
        print("[05단계 결과] 최종 코호트")
        p = "data/cohort/cohort_final.parquet"
        n, pat, pos = con.execute(
            "select count(*), count(distinct subject_id), sum(y::int) from '%s'" % p).fetchone()
        oks += [show("입실 수", n, COHORT_EXPECTED["stays"]), show("환자 수", pat, COHORT_EXPECTED["patients"]),
                show("라벨 1 입실 수", int(pos), COHORT_EXPECTED["positives"])]
        r = con.execute("select sum(aki_by_creat::int), sum(aki_by_uo::int) from '%s'" % p).fetchone()
        print("  참고: 라벨 1 중 크레아티닌 기준 %s건, 소변량 기준 %s건 (두 기준에 모두 해당하는 건 포함. 팀원 보고서: 778 / 1,092 / 둘 다 137)"
              % (f"{int(r[0]):,}", f"{int(r[1]):,}"))
    else:
        print("[11단계 결과] 피처·분할")
        w = "data/features/wide.parquet"
        n = con.execute("select count(*) from '%s'" % w).fetchone()[0]
        cols = len(con.execute("describe select * from '%s'" % w).fetchall())
        oks += [show("wide 행 수", n, WIDE_EXPECTED_ROWS), show("wide 열 수", cols, WIDE_EXPECTED_COLS)]
        for sp, exp in SPLIT_EXPECTED.items():
            got = dict(con.execute("select %s, count(*) from 'data/features/splits.parquet' group by 1" % sp).fetchall())
            for part, e in exp.items():
                key = {"val": ("val", "validation")}.get(part, (part,))
                g = next((got[k] for k in key if k in got), 0)
                oks.append(show("%s %s" % (sp, part), g, e))
    print("\n=> " + ("모두 일치합니다. 다음 단계로 진행해도 됩니다." if all(oks) else
                    "다른 항목이 있습니다. MIMIC 파일이 덜 받아졌거나 버전이 다를 수 있으니 팀원에게 확인한 뒤 진행하세요."))
    sys.exit(0 if all(oks) else 3)


if __name__ == "__main__":
    main()
