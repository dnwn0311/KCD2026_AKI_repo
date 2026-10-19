#!/usr/bin/env bash
# 팀원 전처리 파이프라인(mimic-iv-aki-preprocessing)을 WSL 에서 실행한다.
#
# 하는 일
#   1) team_pipeline/ 코드를 ~/aki_pipeline 으로 복사 (WSL 안 디스크가 빠름)
#   2) 별도 가상환경 ~/aki_venv 만들기 (학습용 ~/kcd_venv 는 건드리지 않음)
#   3) D:\MIMIC 파일을 팀원 코드가 읽는 자리에 연결 (team_link_mimic.py)
#   4) 01~05단계(코호트) 실행 -> team_check.py 로 팀원 보고서 숫자(11,000건, 양성 1,733)와 비교
#   5) 06단계(약물 매핑): data/external/rxnav_cache.json 이 있으면 바로, 없으면 인터넷(RxNav)으로 (최대 1시간)
#   6) 07~12단계 실행 -> 결과의 작은 표만 _work/team/ 으로 복사
#
# 사용 (WSL):
#   bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_team_pipeline.sh"
# 옵션 (환경 변수):
#   FROM=02          해당 단계부터 다시 실행 (예: 01 은 오래 걸리니 이미 끝났으면 FROM=02)
#   STOP_AFTER=05    해당 단계까지만 실행 (코호트 숫자만 먼저 확인하고 싶을 때)
#   MIMIC_ROOT=/mnt/d/MIMIC   MIMIC 폴더
#   AKI_PROJ=~/aki_pipeline   작업 폴더
#   FORCE=1          학습이 도는 중에도 실행 (권장하지 않음: 메모리 부족으로 학습이 죽을 수 있음)
#
# 환자 단위 자료는 이 PC 안에서만 만들어진다. 어디에도 올리지 않는다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMIC="${MIMIC_ROOT:-/mnt/d/MIMIC}"
PROJ="${AKI_PROJ:-$HOME/aki_pipeline}"
VENV="${AKI_VENV:-$HOME/aki_venv}"
FROM="${FROM:-01}"
STOP_AFTER="${STOP_AFTER:-12}"
STEPS=(01_csv_to_parquet 02_build_base_cohort 03_add_esrd_flag 04_build_labels 05_build_final_cohort
       06_build_drug_map 07_build_features 08_build_splits 09_build_feature_dictionary
       10_build_default_features 11_export_csv 12_build_standalone_report)

say() { echo; echo "== $*  ($(date +%H:%M:%S))"; }

# ---- 0. 사전 점검
[ -d "$MIMIC" ] || { echo "[오류] MIMIC 폴더가 없습니다: $MIMIC  (MIMIC_ROOT=... 로 지정)"; exit 1; }
if pgrep -f "train_qlora.py" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  echo "[중단] 처방 모델 학습(train_qlora.py)이 아직 돌고 있습니다."
  echo "  이 파이프라인은 메모리와 디스크를 많이 써서 학습이 중간에 죽을 수 있습니다."
  echo "  학습이 끝난 뒤 다시 실행하세요. (그래도 실행하려면 FORCE=1)"
  exit 1
fi
[ -f "$HERE/team_pipeline/config.yaml" ] || { echo "[오류] $HERE/team_pipeline/ 이 없습니다"; exit 1; }
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "WSL 여유 디스크 ${FREE_GB} GB (변환 결과 약 10~20 GB 필요)"
[ "${FREE_GB:-0}" -ge 40 ] || { echo "[오류] WSL 디스크 여유가 40 GB 미만입니다. 정리 후 다시 실행하세요."; exit 1; }
MEM_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "사용 가능 메모리 ${MEM_GB} GB"
DUCK_MEM="${DUCKDB_MEM:-$(( MEM_GB * 6 / 10 ))GB}"

# ---- 1. 코드 복사 (data/ 는 덮어쓰지 않음)
say "코드를 $PROJ 로 복사"
mkdir -p "$PROJ"
for d in scripts docs; do rm -rf "$PROJ/$d"; cp -r "$HERE/team_pipeline/$d" "$PROJ/$d"; done
cp "$HERE/team_pipeline/config.yaml" "$HERE/team_pipeline/requirements.txt" "$PROJ/"
mkdir -p "$PROJ/data/external"
if [ -f "$HERE/team_pipeline/data/external/rxnav_cache.json" ] && [ ! -f "$PROJ/data/external/rxnav_cache.json" ]; then
  cp "$HERE/team_pipeline/data/external/rxnav_cache.json" "$PROJ/data/external/"
  echo "rxnav_cache.json 복사됨"
fi
# 01단계 변환이 메모리를 다 쓰지 않도록 DuckDB 메모리 상한만 추가 (결과는 같음. 복사본에만 적용)
if ! grep -q "memory_limit" "$PROJ/scripts/01_csv_to_parquet.py"; then
  mkdir -p "$PROJ/data/duckdb_tmp"
  sed -i "s|    con = duckdb.connect()|    con = duckdb.connect()\n    con.execute(\"set memory_limit='${DUCK_MEM}'\")\n    con.execute(\"set temp_directory='${PROJ}/data/duckdb_tmp'\")|" "$PROJ/scripts/01_csv_to_parquet.py"
  echo "01 단계에 DuckDB 메모리 상한 ${DUCK_MEM} 추가 (복사본)"
fi

# ---- 2. 가상환경
if [ ! -x "$VENV/bin/python" ]; then
  say "가상환경 $VENV 만들기"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  # 팀원이 쓴 버전으로 맞춘다. 안 되면 버전 제한 없이
  "$VENV/bin/pip" install -q duckdb==1.5.6 pandas==3.0.6 pyarrow==26.0.0 numpy==2.5.4 pyyaml==6.0.3 requests==2.34.2 \
      scikit-learn==1.9.1 scipy==1.18.1 lightgbm \
    || "$VENV/bin/pip" install -q duckdb pandas pyarrow numpy pyyaml requests scikit-learn scipy lightgbm
fi
PY="$VENV/bin/python"
"$PY" -c "import duckdb, pandas, pyarrow, yaml; print('패키지 확인: duckdb', duckdb.__version__, '| pandas', pandas.__version__)"

# ---- 3. MIMIC 파일 연결
if [ "$FROM" = "01" ]; then
  say "MIMIC 파일을 팀원 코드 위치에 연결"
  "$PY" "$HERE/team_link_mimic.py" --mimic "$MIMIC" --dest "$PROJ/data/mimiciv/3.1"
fi

# ---- 4. 단계 실행
cd "$PROJ"
started=0
for s in "${STEPS[@]}"; do
  n="${s%%_*}"
  [ "$n" = "$FROM" ] && started=1
  [ "$started" = "1" ] || { echo "-- 건너뜀 $s"; continue; }

  if [ "$n" = "06" ]; then
    if [ ! -f data/external/rxnav_cache.json ]; then
      if ! curl -sS -m 15 -o /dev/null "https://rxnav.nlm.nih.gov/REST/version.json" 2>/dev/null; then
        echo
        echo "[멈춤] 06단계(약물 매핑)에 필요한 rxnav_cache.json 이 없고, 인터넷(RxNav)도 연결되지 않습니다."
        echo "  팀원에게 data/external/rxnav_cache.json 파일을 받아 아래 위치에 넣고 FROM=06 으로 다시 실행하세요."
        echo "    $HERE/team_pipeline/data/external/rxnav_cache.json"
        echo "  (코호트 01~05단계 결과는 이미 저장되어 있습니다)"
        exit 4
      fi
      echo "rxnav_cache.json 이 없어 인터넷으로 약물 매핑을 만듭니다. 처음에는 1시간쯤 걸립니다."
    fi
  fi

  say "$s"
  "$PY" -I "scripts/$s.py"

  if [ "$n" = "01" ]; then "$PY" "$HERE/team_check.py" --stage tables --proj "$PROJ" || {
      echo "[중단] 변환된 표의 크기가 보고서와 다릅니다. 위 목록에서 '다름'인 파일을 확인하세요."; exit 3; }; fi
  if [ "$n" = "05" ]; then "$PY" "$HERE/team_check.py" --stage cohort --proj "$PROJ" || {
      [ "${FORCE_CONTINUE:-0}" = "1" ] || { echo "[중단] 코호트 숫자가 팀원 보고서와 다릅니다. (그래도 계속하려면 FORCE_CONTINUE=1)"; exit 3; }; }; fi
  [ "$n" = "$STOP_AFTER" ] && break
done

# ---- 5. 결과 복사
if [ -f data/features/wide.parquet ]; then
  "$PY" "$HERE/team_check.py" --stage final --proj "$PROJ" || true
  say "작은 결과 표를 $HERE/_work/team/ 으로 복사"
  mkdir -p "$HERE/_work/team" "$HERE/results"
  cp data/cohort/cohort_final.parquet data/features/wide.parquet data/features/splits.parquet \
     data/features/default_features.json "$HERE/_work/team/"
  cp data/cohort/flow_base.csv data/cohort/flow_final.csv "$HERE/results/" 2>/dev/null || true
  ls -la "$HERE/_work/team/"
  echo
  echo "다음:  python train_risk_team.py   (위험 모델)   ->   python prep_rx_team.py   (처방 데이터)"
else
  echo; echo "[참고] 코호트 단계까지 끝났습니다 (wide 표는 아직 없음)."
fi
