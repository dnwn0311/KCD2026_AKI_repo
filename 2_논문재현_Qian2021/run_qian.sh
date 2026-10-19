#!/usr/bin/env bash
# Qian 2021 방식 재현 (Linux/WSL). 사용:
#   MIMIC_ROOT=$HOME/aki_pipeline/data/mimiciv/3.1_parquet bash run_qian.sh
#   MIMIC_ROOT   hosp/*.parquet, icu/*.parquet (또는 .csv.gz) 구조의 MIMIC-IV 폴더
#   EXTRA        build 에 더할 옵션. 파일이 흩어져 있으면 예:
#                EXTRA="--patients-file /mnt/d/MIMIC/patients.csv.gz --chartevents-file /mnt/d/MIMIC/chartevents.csv-001-013.gz"
#   SKIP_CNN=1   torch 가 없을 때
set -euo pipefail
cd "$(dirname "$0")"
MIMIC_ROOT="${MIMIC_ROOT:-$HOME/aki_pipeline/data/mimiciv/3.1_parquet}"
EXTRA="${EXTRA:-}"
CNN=""; [ "${SKIP_CNN:-0}" = "1" ] && CNN="--skip-cnn"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8

if [ ! -f qian_table.parquet ]; then
  echo "== build: 코호트·특징 표 만들기 (환자 단위 파일 qian_table.parquet 이 생김, PC 밖으로 내보내지 않기)"
  # shellcheck disable=SC2086
  "$PY" qian2021_replicate.py build --mimic-root "$MIMIC_ROOT" --out qian_table.parquet $EXTRA
else
  echo "== build 생략: qian_table.parquet 이 이미 있음"
fi
echo "== run 1/2: 논문 방식 -> qian_out/"
"$PY" qian2021_replicate.py run --table qian_table.parquet --out qian_out $CNN
echo "== run 2/2: 엄격 조건(보완을 학습에서만 + 1일차 AKI 제외) -> qian_out_strict/"
"$PY" qian2021_replicate.py run --table qian_table.parquet --out qian_out_strict --strict --exclude-day1-aki $CNN
echo "완료. summary.txt 의 [표 2], [표 4], [점검] 블록을 확인하세요."
