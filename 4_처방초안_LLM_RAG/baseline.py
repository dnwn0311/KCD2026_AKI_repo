# -*- coding: utf-8 -*-
"""
3단계: 기준선 성능 (LLM 학습 효과를 비교하기 위한 단순 규칙)

- freq_rule : 신규처방은 상황별로 train 정답에서 가장 흔한 약 k개 (k = 상황별 평균 개수, 반올림, 최소 1),
              감량·증량·중단은 "그 상황에서 그 약이 투여 중일 때 train 에서 해당 변경이 일어난 비율"이
              기준(기본 30%) 이상인 투여 중인 약
- new_only  : 신규처방만 freq_rule 과 같게, 감량·증량·중단은 제안하지 않음 (이전 버전과 같은 방식)
- empty     : 아무것도 제안하지 않음 (참고용)

출력: results/baseline.json (집계만. KCD_DATA_DIR 로 다른 데이터 폴더를 쓰면 baseline_<폴더>.json)
"""
import json
import os
from collections import Counter, defaultdict

import common as C

CHANGE_RATE = 0.30
MIN_ACTIVE = 20  # 비율을 믿을 만큼 투여 중이던 사례가 있어야 함


def read(sp):
    with open(os.path.join(C.DATA, "%s.jsonl" % sp), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    train, test = read("train"), read("test")
    by_s = defaultdict(list)
    act_cnt = defaultdict(Counter)                       # (상황) -> 약이 투여 중이던 사례 수
    chg_cnt = defaultdict(lambda: defaultdict(Counter))  # (상황, 변경) -> 약별 변경 수
    for r in train:
        s = r["scenario"]
        by_s[s].append(r["label"]["신규처방"])
        act_cnt[s].update(set(r.get("active", [])))
        for k in C.ACTIONS[1:]:
            chg_cnt[s][k].update(r["label"][k])
    topk = {}
    for s, labels in by_s.items():
        cnt = Counter(d for L in labels for d in L)
        k = max(1, round(sum(len(L) for L in labels) / len(labels)))
        topk[s] = [d for d, _ in cnt.most_common(k)]
    rules = {s: {k: sorted(d for d, n in chg_cnt[s][k].items()
                           if act_cnt[s][d] >= MIN_ACTIVE and n / act_cnt[s][d] >= CHANGE_RATE)
                 for k in C.ACTIONS[1:]} for s in by_s}

    def freq_rule(r):
        s, act = r["scenario"], set(r.get("active", []))
        out = {"신규처방": [d for d in topk.get(s, []) if d not in act]}
        for k in C.ACTIONS[1:]:
            out[k] = [d for d in rules.get(s, {}).get(k, []) if d in act]
        return out

    def new_only(r):
        out = C.empty_label()
        out["신규처방"] = freq_rule(r)["신규처방"]
        return out

    res = {}
    for name, rule in [("freq_rule", freq_rule), ("new_only", new_only), ("empty", lambda r: C.empty_label())]:
        rows = [(r["scenario"], r["label"], rule(r)) for r in test]
        res[name] = C.set_metrics(rows)
    res["freq_topk_lists"] = topk
    res["change_rules"] = rules
    name = "baseline.json" if os.path.basename(C.DATA) == "data" else "baseline_%s.json" % os.path.basename(C.DATA)
    C.save_json(res, os.path.join(C.RESULTS, name))
    o = res["freq_rule"]["overall"]
    print("기준선(freq_rule) test: 전체 micro F1 %.3f, 신규 %.3f, 감량 %.3f, 증량 %.3f, 중단 %.3f"
          % (o["micro_f1"], *[o["by_action"][k]["f1"] or 0 for k in C.ACTIONS]))


if __name__ == "__main__":
    main()
