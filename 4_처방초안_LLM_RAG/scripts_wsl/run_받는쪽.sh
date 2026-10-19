#!/usr/bin/env bash
# 받는 쪽 실행 순서: 전처리 → 받은 위험 예측 모델로 점수 → 처방 LLM 입력 → 처방 초안 생성
# 사용: MIMIC_ROOT=/경로/MIMIC bash run_받는쪽.sh   (이미 끝난 단계는 SKIP_PREP=1 로 건너뛰기)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
: "${MIMIC_ROOT:?MIMIC_ROOT 를 MIMIC-IV 폴더로 지정하세요 (예: export MIMIC_ROOT=/data/mimiciv)}"
PY="${PY:-python}"
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "[1/5] 원본 추출"; $PY prep_extract.py
  echo "[2/5] AKI 판정, 위험 예측 입력"; $PY prep_aki.py
fi
echo "[3/5] 받은 위험 예측 모델로 점수"; $PY score_risk.py
echo "[4/5] 처방 LLM 입력 생성"; $PY prep_rx.py
echo "[5/5] 처방 초안 생성 (test)"; $PY infer.py --split test "$@"
echo "완료: _work/drafts_test.jsonl (검증 AI 입력), results/drafts_test_metrics.json"
