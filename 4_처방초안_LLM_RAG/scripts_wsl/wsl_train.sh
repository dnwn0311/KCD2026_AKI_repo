#!/usr/bin/env bash
# WSL 에서 처방 LLM 학습 (자동 재시작) → 끝나면 어댑터를 Windows 폴더로 복사 → (선택) 평가
#
# 먼저 Windows 에서 1_전처리.bat, 2_AKI위험도_학습.bat 을 끝내 data/ 를 만들어 두세요.
# 사용 (WSL 터미널에서):
#   bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_train.sh"            학습만
#   DO_EVAL=1 bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_train.sh"  학습 후 평가까지
#
# 빠르게 돌리기 위해 코드와 학습 데이터는 WSL 안(~/kcd_rx)으로 복사해서 씁니다
# (Windows 드라이브 /mnt/c 를 거치면 파일 읽기가 몇 배 느림). 진행 상황 파일(results/)만 Windows 폴더에 바로 기록해
# Windows 쪽에서도 results\train_status_qlora_aki.json 으로 확인할 수 있습니다.
# 학습 중에는 Windows 쪽에서 같은 GPU 를 쓰는 작업을 하지 마세요.
set -uo pipefail
WIN_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/kcd_rx}"
VENV="${VENV:-$HOME/kcd_venv}"
RUN="${RUN:-qlora_aki}"
MAX_TRY="${MAX_TRY:-30}"
RETRY_WAIT="${RETRY_WAIT:-60}"

sync_dir() {  # 폴더 내용을 그대로 복사 (rsync 가 없으면 cp)
  if command -v rsync >/dev/null; then rsync -a --delete "$1/" "$2/"; else rm -rf "$2" && mkdir -p "$2" && cp -r "$1/." "$2/"; fi
}

[ -f "$VENV/bin/activate" ] || { echo "[오류] 가상환경이 없습니다. 먼저 wsl_setup.sh 를 실행하세요."; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ -f "$WIN_DIR/data/train.jsonl" ] || { echo "[오류] $WIN_DIR/data/train.jsonl 이 없습니다. Windows 에서 1_전처리.bat, 2_AKI위험도_학습.bat 을 먼저 실행하세요."; exit 1; }

echo "== 코드와 학습 데이터를 $WORK 로 복사"
mkdir -p "$WORK/data" "$WORK/runs" "$WIN_DIR/results"
cp -u "$WIN_DIR"/*.py "$WORK/"
sync_dir "$WIN_DIR/data" "$WORK/data" || { echo "[오류] 학습 데이터 복사 실패"; exit 1; }
if [ ! -L "$WORK/results" ]; then
  rm -rf "$WORK/results"
  ln -s "$WIN_DIR/results" "$WORK/results"
fi
cd "$WORK"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # Linux 에서는 메모리 조각화를 줄이는 이 설정을 쓸 수 있음
# triton 은 TRITON_OVERRIDE_ARCH 가 설정돼 있으면 "sm120" 형태만 받는다. 형식이 틀린 값이 환경에 남아 있으면 지운다
if [ -n "${TRITON_OVERRIDE_ARCH:-}" ] && ! [[ "$TRITON_OVERRIDE_ARCH" =~ ^sm[0-9]+$ ]]; then
  echo "[참고] TRITON_OVERRIDE_ARCH='$TRITON_OVERRIDE_ARCH' 형식이 맞지 않아 해제합니다"
  unset TRITON_OVERRIDE_ARCH
fi

python - <<'PY'
try:
    import fla  # noqa
    print("선형 어텐션 가속: 사용")
except Exception:
    print("[주의] flash-linear-attention 이 없어 느린 기본 구현으로 학습합니다 (wsl_setup.sh 확인)")
PY

LOG="results/train_stdout_$RUN.log"
SUP="results/supervisor_$RUN.log"
for try in $(seq 1 "$MAX_TRY"); do
  [ -f "results/train_info_$RUN.json" ] && break
  echo "[$(date '+%F %T')] attempt $try start (WSL)" >> "$SUP"
  echo "[$(date '+%F %T')] 학습 시도 $try: runs/$RUN (진행: results/train_status_$RUN.json, 기록: $LOG)"
  python train_qlora.py --out "runs/$RUN" --limit-train 0 --max-len 1536 --batch 1 --grad-accum 16 \
    --save-steps 100 --eval-steps 200 --resume --quiet "$@" >> "$LOG" 2>&1
  code=$?
  echo "[$(date '+%F %T')] attempt $try exit code $code" >> "$SUP"
  [ -f "results/train_info_$RUN.json" ] && break
  echo "학습이 멈췄습니다 (종료 코드 $code). ${RETRY_WAIT}초 뒤 마지막 체크포인트에서 이어서 학습합니다..."
  sleep "$RETRY_WAIT"
done
if [ ! -f "results/train_info_$RUN.json" ]; then
  echo "[$(date '+%F %T')] gave up after $MAX_TRY attempts" >> "$SUP"
  echo "[오류] $MAX_TRY 번 시도 후 중단했습니다. $LOG 를 확인하세요."
  exit 1
fi
echo "[$(date '+%F %T')] finished (WSL)" >> "$SUP"

echo "== 어댑터를 Windows 폴더로 복사: $WIN_DIR/runs/$RUN/adapter"
mkdir -p "$WIN_DIR/runs/$RUN"
sync_dir "runs/$RUN/adapter" "$WIN_DIR/runs/$RUN/adapter" || { echo "[오류] 어댑터 복사 실패 - $WORK/runs/$RUN/adapter 에 있습니다"; exit 1; }

if [ "${DO_EVAL:-0}" = "1" ]; then
  echo "== 평가 (학습한 모델 600건, 학습 전 모델 200건)"
  python eval_model.py --adapter "runs/$RUN/adapter" --tag "$RUN" || echo "[오류] 평가 실패"
  python eval_model.py --no-adapter --tag zeroshot --limit 200 || echo "[오류] 학습 전 모델 평가 실패"
  mkdir -p "$WIN_DIR/_work"
  cp -f _work/preds_*.jsonl "$WIN_DIR/_work/" 2>/dev/null || true
fi
echo "완료. Windows 에서 5_전달패키지.bat 으로 팀원 전달 패키지를 만들 수 있습니다."
