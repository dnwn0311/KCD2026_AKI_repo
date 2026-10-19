# -*- coding: utf-8 -*-
"""
RAG 검증(wsl_run_rag.sh)이 언제 끝나는지 알려 준다. 다른 WSL 창에서 실행 (GPU·모델을 건드리지 않음, 집계 숫자만 출력).
  python3 /mnt/c/Users/User/Documents/KCD_처방추천/rag_progress.py            # 400건(hold400) 기본값
  IDS=<사례 ID 목록> BASE=0 SECS=90 python3 .../rag_progress.py

원리: 판정 결과가 ~/rag_work/rag_verifier_out/llm_cache.jsonl 에 한 줄씩 쌓인다.
  전체 건수  = 대상 사례(IDS)의 초안 항목 수 (위험도 교체가 끝난 rag_handoff_refreshed/drafts.jsonl 에서 셈)
  이미 있던 줄 = BASE (조정용 200건 때 만든 334줄이 같은 캐시 파일에 남아 있음. 처음부터 새 캐시였다면 BASE=0)
  SECS 초 동안 늘어나는 줄 수로 속도를 재고, 남은 시간과 끝나는 시각을 계산한다.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta

HOME = os.path.expanduser("~")
WORK = os.environ.get("WORK", os.path.join(HOME, "rag_work"))
CACHE = os.path.join(WORK, "rag_verifier_out", "llm_cache.jsonl")
DRAFTS = os.environ.get("DRAFTS", os.path.join(WORK, "rag_handoff_refreshed", "drafts.jsonl"))
IDS = os.environ.get("IDS", "/mnt/c/Users/User/Documents/KCD_처방추천/_work/gold_split/hold_ids.txt")
BASE = int(os.environ.get("BASE", "334"))
SECS = int(os.environ.get("SECS", "60"))
ACTIONS = ["신규처방", "감량", "증량", "중단"]


def nlines(p):
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def expected_items():
    ids = {l.strip() for l in open(IDS, encoding="utf-8") if l.strip()}
    n = 0
    with open(DRAFTS, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            c = json.loads(l)
            if c["case_id"] in ids:
                n += sum(len((c.get("draft") or {}).get(k, [])) for k in ACTIONS)
    return n, len(ids)


def main():
    if not os.path.exists(DRAFTS):
        sys.exit("[안내] %s 이 아직 없습니다. 실행 2단계(위험도 교체)가 끝난 뒤 다시 실행하세요." % DRAFTS)
    total, n_cases = expected_items()
    a = nlines(CACHE)
    if a is None:
        sys.exit("[안내] 판정 캐시가 아직 없습니다. 모델을 올리고 첫 묶음이 끝나면 생깁니다. 몇 분 뒤 다시 실행하세요.")
    base = BASE if a >= BASE else 0
    print("대상 %d사례 / 예상 판정 %d건 (이미 있던 캐시 %d줄은 제외)" % (n_cases, total, base))
    t0 = time.time()
    time.sleep(SECS)
    b = nlines(CACHE)
    dt = (time.time() - t0) / 60
    rate = (b - a) / dt if dt > 0 else 0
    done = max(b - base, 0)
    left = max(total - done, 0)
    print("지금까지 %d/%d건 (%.0f%%), 최근 %.1f분간 %d건 → 분당 %.1f건" % (done, total, 100 * done / max(total, 1), dt, b - a, rate))
    if left == 0:
        print("판정은 끝났습니다. 지표 계산·대시보드 파일 쓰기 중이거나 이미 완료되었을 수 있습니다 (화면에 '완료:' 가 나오는지 확인).")
    elif rate <= 0:
        print("그 시간에 늘어난 건이 없습니다. 모델 로딩 중이거나 멈췄을 수 있으니 nvidia-smi 로 GPU 사용률을 확인하세요.")
    else:
        mins = left / rate
        end = datetime.now() + timedelta(minutes=mins)
        print("남은 %d건 → 약 %.0f분 후, %s 쯤 판정 끝 (이후 지표 계산에 1~3분 더)" % (left, mins, end.strftime("%H:%M")))


if __name__ == "__main__":
    main()
