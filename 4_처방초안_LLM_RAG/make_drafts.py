# -*- coding: utf-8 -*-
"""
RAG 검증 담당자에게 넘길 처방 초안 묶음을 만든다 (서버에서 실행)

입력: _work/votes_<tag>_s*.jsonl (eval_sample.py 조각별 투표), data_team/<split>.jsonl, results/tau_<val tag>.json
출력 폴더 _work/rag_handoff_<tag>/  (환자 단위 자료: 팀이 정한 안전한 경로로만 전달)
  drafts.jsonl      사례마다 위험도, 환자 상태 요약(LLM 이 본 입력 그대로), 투여 중인 AKI 관련 약,
                    초안(변경 종류별 약), 초안 약의 득표율, 초안에 못 든 후보 약의 득표율
  drafts.csv        초안 항목 한 줄씩 (case_id, 신호등, 확률, 변경 종류, 약, 득표율) - 엑셀로 보기용
  drug_vocab.json   대상 약 목록과 train 정답 등장 횟수 (집계)
  rag_result_template.csv  RAG 검증 결과를 돌려받을 양식
정답(실제 처방 변경)은 묶음에 넣지 않고 _work/rag_labels_<tag>.jsonl 에 따로 둔다
  -> RAG 결과를 돌려받은 뒤 우리 쪽에서 채점 (검증하는 쪽이 정답을 모르는 상태 유지)

사용:
  python make_drafts.py --tag team27_test --shards 4 --tau results/tau_team27_val.json
"""
import argparse
import csv
import json
import os
from collections import Counter

import common as C
import eval_sample as E

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--shards", type=int, required=True)
ap.add_argument("--tau", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--out-dir", default=None)
a = ap.parse_args()

tau = json.load(open(a.tau, encoding="utf-8"))["tau"]
meta = {}
with open(os.path.join(C.DATA, "%s.jsonl" % a.split), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        meta[r["case_id"]] = r
cases = []
for i in range(a.shards):
    p = os.path.join(C.WORK, "votes_%s_s%d.jsonl" % (a.tag, i))
    if not os.path.exists(p):
        raise SystemExit("[오류] %s 이 없습니다. 모든 조각이 끝났는지 확인하세요." % p)
    with open(p, encoding="utf-8") as f:
        cases += [json.loads(x) for x in f if x.strip()]

# 대상 약 목록 (train 정답에 남은 약 = 어휘)
voc = {k: Counter() for k in C.ACTIONS}
with open(os.path.join(C.DATA, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        lab = json.loads(line)["label"]
        for k in C.ACTIONS:
            voc[k].update(lab.get(k, []))
drugs = sorted(set(d for k in C.ACTIONS for d in voc[k]))

out = a.out_dir or os.path.join(C.WORK, "rag_handoff_%s" % a.tag)
os.makedirs(out, exist_ok=True)
n_items, n_empty, miss, n_cases = 0, 0, 0, 0
tier_cnt = Counter()
with open(os.path.join(out, "drafts.jsonl"), "w", encoding="utf-8") as fj, \
        open(os.path.join(out, "drafts.csv"), "w", encoding="utf-8-sig", newline="") as fc, \
        open(os.path.join(C.WORK, "rag_labels_%s.jsonl" % a.tag), "w", encoding="utf-8") as fl:
    w = csv.writer(fc)
    w.writerow(["case_id", "risk_tier", "risk_prob", "action", "drug", "vote_ratio"])
    for c in sorted(cases, key=lambda x: x["case_id"]):
        m = meta.get(c["case_id"])
        if m is None:
            miss += 1
            continue
        n_cases += 1
        draft = E.make_pred(c["votes"], tau)
        conf = {k: {d: c["votes"][k][d] for d in draft[k]} for k in C.ACTIONS}
        cand = {k: {d: v for d, v in sorted(c["votes"].get(k, {}).items(), key=lambda x: -x[1]) if d not in draft[k]}
                for k in C.ACTIONS}
        rec = {"case_id": c["case_id"], "risk_tier": m.get("tier"), "risk_prob": m.get("prob"),
               "risk_factors": m.get("factors"),
               "patient_summary": m["messages"][1]["content"],
               "active_aki_drugs": m.get("active", []),
               "draft": draft, "vote_ratio": conf, "other_candidates": cand}
        fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fl.write(json.dumps({"case_id": c["case_id"], "label": c["label"]}, ensure_ascii=False) + "\n")
        k_items = 0
        for k in C.ACTIONS:
            for d in draft[k]:
                w.writerow([c["case_id"], m.get("tier"), m.get("prob"), k, d, conf[k][d]])
                k_items += 1
        n_items += k_items
        n_empty += (k_items == 0)
        tier_cnt[m.get("tier")] += 1

C.save_json({"drugs": drugs, "n_drugs": len(drugs),
             "train_label_count": {k: {d: C.suppress(n) for d, n in voc[k].most_common()} for k in C.ACTIONS},
             "note": "train 정답에서 15회 이상 나온 AKI 관련 약. 이름은 MIMIC prescriptions 의 약 이름을 정리한 것"},
            os.path.join(out, "drug_vocab.json"))
with open(os.path.join(out, "rag_result_template.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "action", "drug", "verdict", "modified_action", "modified_drug", "reason", "evidence_source"])
    w.writerow(["(예시) aki_risk-2xxxxxxx-...", "중단", "Vancomycin", "accept", "", "", "eGFR 저하와 신독성 위험", "약물 DB 문서 ID"])
    w.writerow(["(예시) aki_risk-2xxxxxxx-...", "신규처방", "Furosemide", "reject", "", "", "저혈압 상태에서 근거 부족", "약물 DB 문서 ID"])
    w.writerow(["(예시) aki_risk-2xxxxxxx-...", "", "", "add", "감량", "Piperacillin-Tazobactam", "신기능에 맞춘 용량 조절", "약물 DB 문서 ID"])

print("[RAG 전달 묶음] 사례 %d건 (빨강 %d / 노랑 %d / 초록 %d), 초안 항목 %d개 (사례당 %.2f), 변경 없음 %d건, 원본에서 못 찾은 사례 %d건"
      % (n_cases, tier_cnt["red"], tier_cnt["yellow"], tier_cnt["green"], n_items, n_items / max(n_cases, 1), n_empty, miss))
print("  폴더: %s  (drafts.jsonl, drafts.csv, drug_vocab.json, rag_result_template.csv)" % out)
print("  정답(채점용, 전달하지 않음): %s" % os.path.join(C.WORK, "rag_labels_%s.jsonl" % a.tag))
