# -*- coding: utf-8 -*-
"""
RAG 검증 결과를 '안전성' 관점으로 집계한다 (환자 단위 행은 출력하지 않고 집계만 낸다).

배경
  정답(실제 처방 변경)과 일치하는지는 약 라벨 근거로 판단할 수 있는 것이 아니다. 라벨 근거 검증이 하는 일은
  "이 약을 AKI 위험이 있는 환자에게 새로 시작/증량해도 되는가"이다. 그래서 위험 기준을 먼저 정하고(safety_groups.json,
  의료진 확인 전 임시 기준), 그 기준에 비추어 RAG 가 무엇을 막았는지 본다.

계산
  위험 후보  = 위험 약 계열 AND 노출을 늘리는 변경(신규처방·증량)
  보호 항목  = 위험 약 계열 AND 노출을 줄이는 변경(중단·감량)   <- 막으면 안 되는 항목 (오차단)
  막음       = verdict 가 accept 가 아닌 것 (reject·modify)
  주의 이상  = conflict_level 이 caution 또는 contraindicated
  근거 충실도 = 모델이 인용한 근거 문장이 실제 라벨 원문에 있는 비율 (evidence_verbatim)

사용
  python eval_safety.py --dir rag_verifier_out_1015_1401_tune200 [--dir rag_verifier_out_..._hold400] [--groups safety_groups.json]
  -> 각 결과 폴더의 eval/ 에 safety_summary.csv, safety_by_group.csv 를 쓴다 (폴더가 여럿이면 합쳐서 첫 폴더 옆 safety_combined/ 에 쓴다)
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return "%d/%d (%.1f%%, 95%%CI %.1f~%.1f)" % (k, n, 100 * k / n, 100 * lo, 100 * hi)


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="RAG 결과 폴더 (여러 번 지정하면 합침)")
    ap.add_argument("--groups", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety_groups.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.groups, encoding="utf-8"))
    UP, DOWN = set(cfg["exposure_up_actions"]), set(cfg["exposure_down_actions"])
    groups = cfg["groups"]
    second = [x.lower() for x in cfg.get("second_tier_renal_dose", {}).get("drugs", [])]

    def group_of(drug):
        d = (drug or "").lower()
        for g, names in groups.items():
            if any(n in d for n in names):
                return g
        return None

    rows, tier = [], {}
    n_all = n_unlinked = 0
    for d in a.dir:
        det = read_csv(os.path.join(d, "rag_result_detail.csv"))
        n_all += len(det)
        n_unlinked += sum(1 for r in det if r.get("status") == "unlinked")
        rows += [r for r in det if r.get("status") == "judged"]
        for r in read_csv(os.path.join(d, "case_summary.csv")):
            tier[r["case_id"]] = r.get("risk_tier", "?")

    blocked = lambda r: r["verdict"] != "accept"
    caution = lambda r: r.get("conflict_level") in ("caution", "contraindicated")
    for r in rows:
        r["_g"] = group_of(r["drug"])
        r["_up"] = r["action"] in UP

    A = [r for r in rows if r["_g"] and r["_up"]]
    B = [r for r in rows if r["_g"] and r["action"] in DOWN]
    O = [r for r in rows if not r["_g"]]
    S = [r for r in O if r["_up"] and any(s in r["drug"].lower() for s in second)]
    ev_ok = sum(1 for r in rows if str(r.get("evidence_verbatim")) in ("1", "1.0", "True", "true"))

    summ = [
        ("판정 항목 수 / 전체 항목 수 (약 라벨 연결 범위)", fmt(len(rows), n_all)),
        ("위험 후보(위험 약 계열 · 신규처방/증량) 중 RAG 가 막은 비율", fmt(sum(map(blocked, A)), len(A))),
        ("위험 후보 중 '주의 이상' 표시 비율", fmt(sum(map(caution, A)), len(A))),
        ("보호 항목(위험 약 계열 · 중단/감량) 중 오차단", fmt(sum(map(blocked, B)), len(B))),
        ("위험 약 계열이 아닌 항목 중 오차단", fmt(sum(map(blocked, O)), len(O))),
        ("전체 항목 중 '주의 이상' 표시 비율 (구별력 확인용)", fmt(sum(map(caution, rows)), len(rows))),
        ("2차 후보(용량 조정 필요 약, 신규/증량) 중 막은 비율", fmt(sum(map(blocked, S)), len(S))),
        ("근거 문장이 라벨 원문과 일치한 비율", fmt(ev_ok, len(rows))),
    ]
    byt = defaultdict(lambda: [0, 0])
    for r in A:
        t = tier.get(r["case_id"], "?")
        byt[t][0] += 1
        byt[t][1] += blocked(r)
    for t in ("red", "yellow", "green"):
        if t in byt:
            summ.append(("위험 후보 중 막은 비율 - 환자 신호등 %s" % t, fmt(byt[t][1], byt[t][0])))

    tab = defaultdict(list)
    for r in rows:
        if r["_g"]:
            tab[(r["_g"], "노출 늘림(신규·증량)" if r["_up"] else "노출 줄임(중단·감량)")].append(r)
    by_group = [["약 계열", "방향", "항목 수", "막음", "막은 비율(%)", "주의 이상", "주의 이상 비율(%)"]]
    for (g, dr), v in sorted(tab.items()):
        nb, nc = sum(map(blocked, v)), sum(map(caution, v))
        by_group.append([g, dr, len(v), nb, round(100 * nb / len(v), 1), nc, round(100 * nc / len(v), 1)])

    outs = [os.path.join(d, "eval") for d in a.dir] if len(a.dir) == 1 else [os.path.join(os.path.dirname(os.path.abspath(a.dir[0])), "safety_combined")]
    for o in outs:
        os.makedirs(o, exist_ok=True)
        with open(os.path.join(o, "safety_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["지표", "값"])
            w.writerows(summ)
        with open(os.path.join(o, "safety_by_group.csv"), "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(by_group)

    print("[안전성 집계] 결과 폴더 %d개, 판정 항목 %d개  (위험 기준: %s, 임시)" % (len(a.dir), len(rows), os.path.basename(a.groups)))
    for k, v in summ:
        print("  %-52s %s" % (k, v))
    print("\n  [약 계열 x 방향]")
    for r in by_group[1:]:
        print("   ", r)
    print("\n  저장:", ", ".join(outs))


if __name__ == "__main__":
    main()
