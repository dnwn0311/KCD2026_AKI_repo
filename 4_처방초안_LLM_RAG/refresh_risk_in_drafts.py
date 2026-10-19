# -*- coding: utf-8 -*-
"""
처방 초안 묶음(drafts.jsonl)의 위험도·신호등·주요 요인을 새 위험 모델 값으로 바꾼다.

배경
  초안은 약물 플래그를 고치기 전 위험 모델(AUROC 0.748)의 확률·신호등·주요 요인을 입력으로 만들었다.
  최종 위험 모델은 약물 플래그를 고친 LightGBM + 추세 모델(runs/aki_risk_team_trend, test AUROC 0.775)이다.
  초안(처방 LLM 출력)과 득표율은 그대로 두고, 화면·검증에 쓰는 위험 정보만 새 값으로 맞춘다.

바뀌는 것
  risk_prob, risk_tier, risk_factors            -> 새 모델 값 (이전 값은 *_old 로 남김)
  patient_summary 의 "[AKI 위험 예측]" 줄, "[위험을 올린 주요 요인]" 줄 -> 새 값으로 교체
  patient_summary_llm_input                     -> 처방 LLM 이 실제로 본 원래 입력 (감사용으로 보관)
  risk_refreshed                                -> 새 값으로 바꿨는지 (True/False)
바뀌지 않는 것
  draft, vote_ratio, other_candidates, active_aki_drugs, case_id, 그 밖의 patient_summary 내용

두 단계 (환자 단위 자료이므로 PC 와 서버 사이는 팀이 정한 안전한 경로로만 옮길 것)
  1) map   : 새 위험 점수 -> case_id 별 표 (risk_refresh_test.jsonl).  WSL, ~/kcd_rx 에서
       python refresh_risk_in_drafts.py map --team-dir _work/team_fixed --out risk_refresh_test.jsonl
  2) apply : drafts.jsonl 에 적용.  drafts.jsonl 이 있는 곳(서버)에서
       python refresh_risk_in_drafts.py apply --drafts <rag_handoff 폴더>/drafts.jsonl --refresh risk_refresh_test.jsonl \\
              --out-dir <새 폴더>
     -> drafts.jsonl, drafts.csv (항목 한 줄씩), refresh_report.json (집계만)
  원본 drafts.jsonl 은 건드리지 않는다.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

TIER_KR = {"red": "빨강(고위험)", "yellow": "노랑(주의)", "green": "초록(낮음)"}   # common.TIER_KR 와 같음
ACTIONS = ["신규처방", "감량", "증량", "중단"]
RISK_LINE = re.compile(r"^\[AKI 위험 예측\].*$", re.M)
FACT_LINE = re.compile(r"^\[위험을 올린 주요 요인\].*$", re.M)
DRUG_WORDS = re.compile(r"NSAID|비스테로이드|소염진통|ACE|ARB|안지오텐신|M01A|C09", re.I)


def case_id(hadm_id, t):
    import pandas as pd
    return "aki_risk-%d-%s" % (int(hadm_id), pd.Timestamp(t).strftime("%Y%m%d%H%M"))


def cmd_map(a):
    import pandas as pd
    td = a.team_dir
    coh = pd.read_parquet(os.path.join(td, "cohort_final.parquet"), columns=["stay_id", "hadm_id", "time_zero"])
    sc = pd.read_parquet(os.path.join(td, a.scores), columns=["stay_id", "split", "prob", "tier", "factors"])
    sc = sc[sc["split"].astype(str) == a.split]
    m = coh.merge(sc, on="stay_id", how="inner")
    m["case_id"] = [case_id(h, t) for h, t in zip(m["hadm_id"], pd.to_datetime(m["time_zero"]))]
    dup = int(m["case_id"].duplicated().sum())
    if dup:
        print("[주의] case_id 가 겹치는 행 %d건 (같은 입원·같은 시각) - 앞의 것만 씁니다" % dup)
        m = m.drop_duplicates("case_id")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in m.itertuples(index=False):
            f.write(json.dumps({"case_id": r.case_id, "prob": round(float(r.prob), 4), "tier": str(r.tier),
                                "factors": "" if r.factors is None or r.factors != r.factors else str(r.factors),
                                "source": "%s/%s" % (os.path.basename(os.path.normpath(td)), a.scores)},
                               ensure_ascii=False) + "\n")
    print("[map] %s 분할 %d건 -> %s" % (a.split, len(m), a.out))
    print("  신호등:", dict(Counter(m["tier"].astype(str))))


def risk_lines(prob, tier, factors):
    head = "[AKI 위험 예측] 향후 24시간 안 AKI 발생 확률 %.0f%%, 신호등 %s" % (100 * float(prob), TIER_KR.get(tier, tier))
    fac = ("[위험을 올린 주요 요인] " + factors) if factors else None
    return head, fac


def swap_summary(summary, prob, tier, factors):
    """patient_summary 의 위험 줄을 새 값으로 바꾼다. 줄이 없으면 맨 앞에 넣는다."""
    head, fac = risk_lines(prob, tier, factors)
    s = summary or ""
    if RISK_LINE.search(s):
        s = RISK_LINE.sub(lambda _: head, s, count=1)
    else:
        s = head + ("\n" + s if s else "")
    if FACT_LINE.search(s):
        s = FACT_LINE.sub(lambda _: fac, s, count=1) if fac else FACT_LINE.sub("", s, count=1).replace("\n\n", "\n")
    elif fac:
        s = s.replace(head, head + "\n" + fac, 1)
    return s


def cmd_apply(a):
    new = {}
    with open(a.refresh, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                new[r["case_id"]] = r
    os.makedirs(a.out_dir, exist_ok=True)
    n, n_ok = 0, 0
    migr = Counter()
    drug_old = drug_new = 0
    with open(a.drafts, encoding="utf-8") as f, \
            open(os.path.join(a.out_dir, "drafts.jsonl"), "w", encoding="utf-8") as fj, \
            open(os.path.join(a.out_dir, "drafts.csv"), "w", encoding="utf-8-sig", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["case_id", "risk_tier", "risk_prob", "action", "drug", "vote_ratio"])
        for l in f:
            if not l.strip():
                continue
            c = json.loads(l)
            n += 1
            old = (c.get("risk_prob"), c.get("risk_tier"), c.get("risk_factors"))
            r = new.get(c["case_id"])
            c["patient_summary_llm_input"] = c.get("patient_summary")
            c["risk_prob_old"], c["risk_tier_old"], c["risk_factors_old"] = old
            if r is not None:
                n_ok += 1
                c["risk_prob"], c["risk_tier"], c["risk_factors"] = r["prob"], r["tier"], r["factors"]
                c["patient_summary"] = swap_summary(c.get("patient_summary"), r["prob"], r["tier"], r["factors"])
                c["risk_refreshed"] = True
                migr[(old[1], r["tier"])] += 1
                drug_old += bool(DRUG_WORDS.search(old[2] or ""))
                drug_new += bool(DRUG_WORDS.search(r["factors"] or ""))
            else:
                c["risk_refreshed"] = False
            fj.write(json.dumps(c, ensure_ascii=False) + "\n")
            for k in ACTIONS:
                for d in (c.get("draft") or {}).get(k, []):
                    w.writerow([c["case_id"], c.get("risk_tier"), c.get("risk_prob"), k, d,
                                (c.get("vote_ratio") or {}).get(k, {}).get(d)])
    rep = {"cases": n, "refreshed": n_ok, "not_found": n - n_ok,
           "tier_old_to_new": {"%s->%s" % k: v for k, v in sorted(migr.items(), key=lambda x: str(x[0]))},
           "factors_mentioning_nsaid_or_acearb": {"old": drug_old, "new": drug_new}}
    with open(os.path.join(a.out_dir, "refresh_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("[apply] 사례 %d건 중 새 값으로 교체 %d건, 못 찾은 사례 %d건 (원래 값 유지, risk_refreshed=False)" % (n, n_ok, n - n_ok))
    print("  신호등 이동(이전->새):", rep["tier_old_to_new"])
    print("  주요 요인에 NSAID/ACEi·ARB 관련 문구가 든 사례: 이전 %d -> 새 %d" % (drug_old, drug_new))
    print("  저장:", os.path.join(a.out_dir, "drafts.jsonl"), "/ drafts.csv / refresh_report.json")
    if n - n_ok:
        print("[주의] 못 찾은 사례가 있습니다. map 단계의 --split, --team-dir 가 초안 사례와 맞는지 확인하세요.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("--team-dir", default="_work/team_fixed")
    m.add_argument("--scores", default="risk_scores_trend.parquet")
    m.add_argument("--split", default="test")
    m.add_argument("--out", default="risk_refresh_test.jsonl")
    p = sub.add_parser("apply")
    p.add_argument("--drafts", required=True)
    p.add_argument("--refresh", required=True)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "apply" and os.path.abspath(os.path.dirname(a.drafts)) == os.path.abspath(a.out_dir):
        sys.exit("[오류] --out-dir 이 원본 drafts.jsonl 과 같은 폴더입니다. 다른 폴더를 지정하세요 (원본을 덮어쓰지 않음).")
    (cmd_map if a.cmd == "map" else cmd_apply)(a)


if __name__ == "__main__":
    main()
