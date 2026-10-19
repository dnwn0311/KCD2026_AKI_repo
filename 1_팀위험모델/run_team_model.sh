#!/usr/bin/env bash
# 팀 위험 모델 전체 실행 (Linux/WSL). 사용:  PIPE=$HOME/aki_pipeline bash run_team_model.sh
#   PIPE   팀원 전처리(mimic-iv-aki-preprocessing) 폴더. 아래 파일이 있어야 한다.
#          data/cohort/cohort_final.parquet, data/features/{wide,splits,hourly}.parquet, data/features/default_features.json
#   STEPS  돌릴 단계 (기본: "copy patch trend models contrib values catalog")
# 끝난 단계를 다시 돌려도 안전하다 (결과 파일을 덮어쓸 뿐).
set -euo pipefail
cd "$(dirname "$0")"
PIPE="${PIPE:-$HOME/aki_pipeline}"
STEPS="${STEPS:-copy patch trend models contrib values catalog}"
PY="${PY:-python}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo; echo "== [$(date '+%m-%d %H:%M')] $*"; }
export PYTHONIOENCODING=utf-8

if has copy; then
  say "0) 팀원 전처리 파일 복사 -> _work/team/"
  mkdir -p _work/team
  for f in data/cohort/cohort_final.parquet data/features/wide.parquet data/features/splits.parquet data/features/default_features.json; do
    [ -f "$PIPE/$f" ] || { echo "[오류] $PIPE/$f 가 없습니다"; exit 1; }
    cp "$PIPE/$f" _work/team/
  done
fi
if has patch; then
  say "1) 약물 플래그 수정본 -> _work/team_fixed/  (duckdb 필요)"
  "$PY" patch_drug_flags.py --proj "$PIPE"
fi
if has trend; then
  say "2) 추세 특징 생성 + 기준 vs +추세 학습 + 확정 모델 저장"
  "$PY" train_risk_trend.py --team-dir _work/team_fixed --hourly "$PIPE/data/features/hourly.parquet" --save
fi
if has models; then
  say "3) 알고리즘 비교 (LightGBM / XGBoost / RF / LR, 시드 3개)"
  "$PY" compare_models.py
fi
if has contrib; then
  say "4) 변수 기여도"
  "$PY" feature_contrib.py
fi
if has values; then
  say "5a) 상위 변수 값 예시"
  "$PY" feature_values.py
fi
if has catalog; then
  say "5b) 입력 변수 전체 값 목록"
  "$PY" feature_catalog.py
fi
say "완료. 집계 결과: results/ , feature_contrib/"
