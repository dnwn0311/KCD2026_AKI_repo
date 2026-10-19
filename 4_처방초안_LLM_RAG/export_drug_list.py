"""위험 모델에 들어간 약물군 목록 내보내기 (의료진 확인용)

각 약물 플래그(신장 관련 10군 + ATC 3단계 분류)마다
  - ATC 코드와 분류명(WHO ATC 3단계, 이 스크립트 안의 표)
  - 플래그가 1인 입실 수와 비율
  - 그 분류로 잡힌 처방의 성분명 상위 목록 (입실 수 기준)
를 CSV 한 장으로 만든다.

성분명 목록에 대한 주의
  drug_map 은 처방 한 건에 성분과 ATC 코드를 묶음으로만 저장한다. 복합제(예: piperacillin + tazobactam)는 성분이
  모두 그 처방의 ATC 코드 아래에 같이 잡히므로, 한 분류의 성분 목록에 다른 분류의 성분이 섞여 보일 수 있다.
  단일 성분 약은 정확하다.

사용 (WSL, 팀원 전처리 환경에서 실행 - duckdb 가 있는 aki_venv):
  cd "/mnt/c/Users/User/Documents/KCD_처방추천"
  ~/aki_venv/bin/python export_drug_list.py
  -> results/drug_list_for_clinicians.csv (엑셀에서 열림, UTF-8 BOM)

환자 단위 자료는 이 PC 안에서만 읽는다. CSV 에는 성분명과 입실 수 집계만 들어간다.
"""
import argparse
import csv
import os
import sys

import duckdb

from patch_drug_flags import ACE_ARB, LOCAL as FIX_LOCAL, NSAID   # 성분명으로 다시 정의한 두 약물군

# 신장 관련 10군 (team_pipeline/config.yaml 의 drugs.kidney_relevant 와 같아야 한다)
KIDNEY = {
    "aminoglycoside": dict(label="Aminoglycoside antibiotics", atc=["J01G"], iv=True, im=True),
    "vancomycin": dict(label="Glycopeptide antibacterials (vancomycin)", atc=["J01XA"], iv=True),
    "pip_tazo": dict(label="Piperacillin (piperacillin-tazobactam)", ing=["piperacillin"], iv=True),
    "polymyxin": dict(label="Polymyxins (colistin)", atc=["J01XB"], iv=True),
    "amphotericin": dict(label="Amphotericin B", atc=["J02AA"], iv=True),
    "nucleoside_antiviral": dict(label="Nucleoside antivirals (acyclovir, ganciclovir)", atc=["J05AB"], iv=True),
    "nsaid": dict(label="NSAIDs", atc=["M01A"]),
    "loop_diuretic": dict(label="Loop diuretics", atc=["C03C"]),
    "ace_arb": dict(label="ACE inhibitors and ARBs", atc=["C09"]),
    "calcineurin": dict(label="Calcineurin inhibitors (tacrolimus, cyclosporine)", atc=["L04AD"]),
}

# WHO ATC 3단계 분류명 (플래그가 만들어질 수 있는 코드). 표에 없는 코드는 빈 칸으로 둔다.
ATC3_NAME = {
    "A01A": "Stomatological preparations", "A02A": "Antacids", "A02B": "Drugs for peptic ulcer and GORD (PPI, H2 blockers)",
    "A03B": "Belladonna and derivatives, plain", "A03F": "Propulsives", "A04A": "Antiemetics and antinauseants",
    "A06A": "Drugs for constipation", "A07A": "Intestinal antiinfectives", "A07D": "Antipropulsives",
    "A07E": "Intestinal antiinflammatory agents", "A08A": "Antiobesity preparations", "A10A": "Insulins and analogues",
    "A10B": "Blood glucose lowering drugs, excl. insulins", "A11C": "Vitamin A and D", "A11D": "Vitamin B1",
    "A11G": "Ascorbic acid (vitamin C)", "A11H": "Other plain vitamin preparations", "A12A": "Calcium",
    "A12B": "Potassium", "A12C": "Other mineral supplements", "B01A": "Antithrombotic agents",
    "B02B": "Vitamin K and other hemostatics", "B03A": "Iron preparations", "B03B": "Vitamin B12 and folic acid",
    "B05B": "I.v. solutions", "B05C": "Irrigating solutions", "B05X": "I.v. solution additives",
    "C01A": "Cardiac glycosides", "C01B": "Antiarrhythmics, class I and III", "C01C": "Cardiac stimulants excl. cardiac glycosides",
    "C01D": "Vasodilators used in cardiac diseases", "C01E": "Other cardiac preparations", "C02A": "Antiadrenergic agents, centrally acting",
    "C02D": "Arteriolar smooth muscle, agents acting on", "C03A": "Low-ceiling diuretics, thiazides",
    "C03B": "Low-ceiling diuretics, excl. thiazides", "C03C": "High-ceiling diuretics (loop)", "C03D": "Potassium-sparing agents",
    "C05A": "Agents for hemorrhoids and anal fissures, topical", "C05B": "Antivaricose therapy", "C07A": "Beta blocking agents",
    "C07F": "Beta blocking agents, other combinations", "C08C": "Calcium channel blockers, vascular selective",
    "C08D": "Calcium channel blockers, cardiac selective", "C09A": "ACE inhibitors, plain", "C09B": "ACE inhibitors, combinations",
    "C09C": "ARBs, plain", "C09D": "ARBs, combinations", "C09X": "Other agents acting on the renin-angiotensin system",
    "C10A": "Lipid modifying agents, plain", "C10B": "Lipid modifying agents, combinations",
    "D01A": "Antifungals for topical use", "D04A": "Antipruritics (topical)", "D06A": "Antibiotics for topical use",
    "D07A": "Corticosteroids, plain (topical)", "D07X": "Corticosteroids, other combinations (topical)",
    "D10A": "Anti-acne preparations for topical use", "D11A": "Other dermatological preparations",
    "G01A": "Gynecological antiinfectives and antiseptics", "G02C": "Other gynecologicals", "G04B": "Other urologicals",
    "G04C": "Drugs used in benign prostatic hypertrophy", "H01B": "Posterior pituitary lobe hormones (vasopressin)",
    "H01C": "Hypothalamic hormones", "H02A": "Corticosteroids for systemic use, plain", "H03A": "Thyroid preparations",
    "H04A": "Glycogenolytic hormones", "J01A": "Tetracyclines", "J01C": "Beta-lactam antibacterials, penicillins",
    "J01D": "Other beta-lactam antibacterials (cephalosporins, carbapenems)", "J01E": "Sulfonamides and trimethoprim",
    "J01F": "Macrolides, lincosamides and streptogramins", "J01G": "Aminoglycoside antibacterials", "J01M": "Quinolone antibacterials",
    "J01R": "Combinations of antibacterials", "J01X": "Other antibacterials (vancomycin, linezolid, colistin ...)",
    "J02A": "Antimycotics for systemic use", "J05A": "Direct acting antivirals", "L04A": "Immunosuppressants",
    "M01A": "Antiinflammatory and antirheumatic products, non-steroids (NSAIDs)", "M02A": "Topical products for joint and muscular pain",
    "M03A": "Muscle relaxants, peripherally acting", "M03B": "Muscle relaxants, centrally acting", "M04A": "Antigout preparations",
    "N01A": "Anesthetics, general", "N01B": "Anesthetics, local", "N02A": "Opioids", "N02B": "Other analgesics and antipyretics",
    "N02C": "Antimigraine preparations", "N03A": "Antiepileptics", "N04B": "Dopaminergic agents", "N05A": "Antipsychotics",
    "N05B": "Anxiolytics", "N05C": "Hypnotics and sedatives", "N06A": "Antidepressants",
    "N06B": "Psychostimulants, agents used for ADHD and nootropics", "N06D": "Anti-dementia drugs", "N07A": "Parasympathomimetics",
    "N07B": "Drugs used in addictive disorders", "P01A": "Agents against amoebiasis and other protozoal diseases",
    "P01B": "Antimalarials", "R01A": "Decongestants and other nasal preparations for topical use",
    "R01B": "Nasal decongestants for systemic use", "R02A": "Throat preparations", "R03A": "Adrenergics, inhalants",
    "R03B": "Other drugs for obstructive airway diseases, inhalants", "R03C": "Adrenergics for systemic use",
    "R03D": "Other systemic drugs for obstructive airway diseases", "R05C": "Expectorants", "R05D": "Cough suppressants",
    "R06A": "Antihistamines for systemic use", "S01A": "Antiinfectives (ophthalmological)",
    "S01B": "Antiinflammatory agents (ophthalmological)", "S01C": "Antiinflammatory agents and antiinfectives in combination (ophthalmological)",
    "S01E": "Antiglaucoma preparations and miotics", "S01G": "Decongestants and antiallergics (ophthalmological)",
    "S01H": "Local anesthetics (ophthalmological)", "S02A": "Antiinfectives (otological)", "S02D": "Other otologicals",
    "V03A": "All other therapeutic products (antidotes etc.)", "V04C": "Other diagnostic agents", "V06D": "Other nutrients",
}

LOCAL = "EYE|EAR|^TP$|^OU$|^OD$|^OS$|INTRAVITREAL|^IVT$"   # 07_build_features.py 와 같은 경로 규칙
ORAL = "^PO|^NG$|ORAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=os.path.expanduser("~/aki_pipeline"))
    ap.add_argument("--out", default="results/drug_list_for_clinicians.csv")
    ap.add_argument("--wide", default=None, help="약물 플래그가 든 wide.parquet (기본: _work/team_fixed/wide.parquet 이 있으면 그것, 없으면 --proj 의 것)")
    ap.add_argument("--top", type=int, default=10, help="분류마다 보여줄 성분 수")
    a = ap.parse_args()

    fixed_wide = os.path.join("_work", "team_fixed", "wide.parquet")
    wide = a.wide or (fixed_wide if os.path.exists(fixed_wide) else os.path.join(a.proj, "data/features/wide.parquet"))
    print("플래그 출처:", wide)
    rx = os.path.join(a.proj, "data/mimiciv/3.1_parquet/hosp/prescriptions.parquet")
    dmap = os.path.join(a.proj, "data/external/drug_map.parquet")
    for p in (wide, rx, dmap):
        if not os.path.exists(p):
            sys.exit("[오류] %s 가 없습니다" % p)

    con = duckdb.connect()
    cols = [r[0] for r in con.execute("describe select * from '%s'" % wide).fetchall()]
    n_stays = con.execute("select count(*) from '%s'" % wide).fetchone()[0]
    print("코호트 입실 %d건, 약물 열 %d개" % (n_stays, sum(c.startswith(("atc_", "atc3_", "drug_")) for c in cols)))

    # 07_build_features.py 와 같은 방식: time zero 전에 시작했고 ICU 입실 전에 끝나지 않은 처방
    con.execute("""
create temp table c as select stay_id, hadm_id, intime, time_zero from '%s'""" % wide)
    con.execute("""
create temp table rx_orders as
select c.stay_id, lower(p.drug) drug, upper(p.route) route,
  case when p.ndc is null or p.ndc in ('0', '') then null else p.ndc end ndc
from c join '%s' p on p.hadm_id = c.hadm_id
where p.starttime < c.time_zero and coalesce(p.stoptime, p.starttime) >= c.intime""" % rx)
    con.execute("""
create temp table rx_m as
select o.stay_id, o.route,
  coalesce(nullif(mn.atc_codes, ''), nullif(mm.atc_codes, ''), '') codes,
  lower(coalesce(nullif(mn.ingredients, ''), nullif(mm.ingredients, ''), '')) ings
from rx_orders o
left join '%s' mn on mn.kind = 'ndc' and mn.key = o.ndc
left join '%s' mm on mm.kind = 'name' and mm.key = o.drug""" % (dmap, dmap))
    # 처방 x 코드 x 성분 (경로 규칙 적용)
    con.execute("""
create temp table rci as
select distinct stay_id, code, ing,
  regexp_matches(coalesce(route, ''), '^IV($|[ _-])') as is_iv, coalesce(route, '') = 'IM' as is_im
from (select stay_id, route, code, unnest(string_split(ings, ';')) ing
      from (select stay_id, route, ings, unnest(string_split(codes, ';')) code from rx_m where codes <> '' and ings <> ''))
where code <> '' and ing <> '' and (
  case when regexp_matches(coalesce(route, ''), '%s') then left(code, 1) in ('D', 'S')
       else left(code, 1) not in ('D', 'S') and (left(code, 3) <> 'A07' or regexp_matches(coalesce(route, ''), '%s')) end)""" % (LOCAL, ORAL))
    # 성분 이름만 필요한 분류(pip_tazo)는 ATC 코드가 없어도 잡아야 하므로 따로
    con.execute("""
create temp table ri as
select distinct stay_id, route, ing,
  regexp_matches(coalesce(route, ''), '^IV($|[ _-])') as is_iv, coalesce(route, '') = 'IM' as is_im
from (select stay_id, route, unnest(string_split(ings, ';')) ing from rx_m where ings <> '') where ing <> ''""")
    fixed_ing = {"atc_M01A_nsaid": NSAID, "atc_C09_ace_arb": ACE_ARB, "atc3_M01A": NSAID}

    def flag_n(col):
        if col not in cols:
            return None
        return con.execute("select sum(%s::int) from '%s'" % (col, wide)).fetchone()[0]

    def top_ings(where_sql, table):
        r = con.execute("select ing, count(distinct stay_id) n from %s where %s group by 1 order by 2 desc, 1 limit %d"
                        % (table, where_sql, a.top)).fetchall()
        return "; ".join("%s (%d)" % (i, n) for i, n in r)

    rows = []
    for key, d in KIDNEY.items():
        route = ""
        if d.get("iv"):
            route = " and (is_iv%s)" % (" or is_im" if d.get("im") else "")
        if key in ("nsaid", "ace_arb"):
            col = "atc_%s_%s" % (d["atc"][0], key)
            lst = ",".join("'%s'" % i for i in fixed_ing[col])
            ings = top_ings("ing in (%s) and not regexp_matches(coalesce(route, ''), '%s')" % (lst, FIX_LOCAL), "ri")
            code_txt = "성분명 목록 기준 (" + d["atc"][0] + " 계열)"
            n = flag_n(col)
            rows.append(["신장 관련 10군", col, code_txt, d["label"], "국소·흡입 경로 제외",
                         n if n is not None else "", ("%.1f" % (100 * n / n_stays)) if n is not None else "", ings])
            continue
        if "atc" in d:
            col = "atc_%s_%s" % (d["atc"][0], key)
            cond = "(" + " or ".join("code like '%s%%'" % p for p in d["atc"]) + ")" + route
            ings, code_txt = top_ings(cond, "rci"), "/".join(d["atc"])
        else:
            col = "drug_%s" % key
            cond = "(" + " or ".join("ing like '%%%s%%'" % i for i in d["ing"]) + ")" + route
            ings, code_txt = top_ings(cond, "ri"), "성분명 포함: " + ", ".join(d["ing"])
        n = flag_n(col)
        rows.append(["신장 관련 10군", col, code_txt, d["label"], "정맥(IV)" + ("·근육(IM)" if d.get("im") else "") + " 처방만" if d.get("iv") else "모든 경로",
                     n if n is not None else "", ("%.1f" % (100 * n / n_stays)) if n is not None else "", ings])

    atc3 = sorted(c[5:] for c in cols if c.startswith("atc3_"))
    for k in atc3:
        n = flag_n("atc3_" + k)
        if k == "M01A":   # NSAID 와 같은 성분명 정의
            lst = ",".join("'%s'" % i for i in NSAID)
            ings, route_txt = top_ings("ing in (%s) and not regexp_matches(coalesce(route, ''), '%s')" % (lst, FIX_LOCAL), "ri"), "성분명 목록 기준 (NSAID 와 동일)"
        else:
            ings, route_txt = top_ings("code like '%s%%'" % k, "rci"), "거친 지표: 복합제·보조 적응증 분류도 켜짐 (피부·눈·귀 코드 규칙 적용)"
        rows.append(["ATC 3단계", "atc3_" + k, k, ATC3_NAME.get(k, ""), route_txt,
                     n if n is not None else "", ("%.1f" % (100 * n / n_stays)) if n is not None else "", ings])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["구분", "모델 열 이름", "ATC 코드", "분류명", "인정하는 처방 경로", "플래그 1인 입실 수", "비율(%)",
                    "성분명 상위 (입실 수) - 복합제는 다른 분류 성분이 섞일 수 있음"])
        w.writerows(rows)
    print("저장: %s (신장 관련 %d군 + ATC 3단계 %d개)" % (a.out, len(KIDNEY), len(atc3)))
    for r in rows[:10]:
        print("  %-26s %-8s 입실 %s건 (%s%%) | %s" % (r[1], r[2], r[5], r[6], r[7][:70]))


if __name__ == "__main__":
    main()
