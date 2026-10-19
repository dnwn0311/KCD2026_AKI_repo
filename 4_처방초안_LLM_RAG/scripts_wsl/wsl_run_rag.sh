#!/usr/bin/env bash
# 처방 초안 RAG 검증 실행 (WSL). 위험도 갱신 -> 노트북 실행 -> 결과를 Windows 폴더로 복사.
#   점검(앞 10건만):  MAX_CASES=10 bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_run_rag.sh"
#   전체:             bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_run_rag.sh"
# 환경 변수(필요할 때만):
#   GPG=<drafts 암호화 파일>   DRAFTS_IN=<이미 푼 drafts.jsonl>   REFRESH=<risk_refresh_test.jsonl>
#   GOLD_PATH=<정답 jsonl|csv>   TUNE_CASES_PATH=<조정에 쓴 사례 ID 목록(평가에서 제외)>
#   CASE_IDS_PATH=<판정할 사례 ID 목록(한 줄에 하나)>   RUN_TAG=<결과 폴더 이름에 붙일 표시>   PROMPT_VERSION=1|2
#   PHARM_DIR / PHARM_GLOB (약물 DB zip 위치)   BATCH_SIZE (메모리 부족이면 2)   RUN_SYNTHESIS=0 (SOAP 종합 생략)
set -e
WIN="/mnt/c/Users/User/Documents/KCD_처방추천"
DL="/mnt/c/Users/User/Downloads"
VENV="${VENV:-$HOME/kcd_venv}"
WORK="$HOME/rag_work${MAX_CASES:+_smoke$MAX_CASES}"
GPG="${GPG:-$DL/rag_handoff_team27_test.tar.gz.gpg}"
REFRESH="${REFRESH:-$WIN/risk_refresh_test.jsonl}"
[ -f "$VENV/bin/activate" ] || { echo "[오류] $VENV 가 없음. VENV=... 로 환경을 지정하거나 wsl_rag_setup.sh 를 먼저 실행하세요."; exit 1; }
if pgrep -f "train_qlora.py|eval_sample.py" >/dev/null; then echo "[중단] 다른 학습·평가가 GPU 를 쓰고 있습니다. 끝난 뒤 다시 실행하세요."; exit 1; fi
source "$VENV/bin/activate"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${TRITON_OVERRIDE_ARCH:-}" ] && ! [[ "$TRITON_OVERRIDE_ARCH" =~ ^sm[0-9]+$ ]]; then unset TRITON_OVERRIDE_ARCH; fi
mkdir -p "$WORK"; cd "$WORK"

echo "== 1. 처방 초안 준비"
if [ -n "$DRAFTS_IN" ]; then
  SRC="$DRAFTS_IN"
else
  SRC="$(find rag_handoff -name drafts.jsonl 2>/dev/null | head -1)"
  if [ -z "$SRC" ]; then
    [ -f "$GPG" ] || { echo "[오류] drafts 암호화 파일이 없음: $GPG  (GPG=... 로 위치 지정, 또는 DRAFTS_IN=... 로 푼 파일 지정)"; exit 1; }
    export GPG_TTY="$(tty)"
    echo "   암호를 입력하세요 (화면에 표시되지 않음)"
    gpg --pinentry-mode loopback -o rag_handoff.tar.gz -d "$GPG"
    mkdir -p rag_handoff && tar -xzf rag_handoff.tar.gz -C rag_handoff && rm -f rag_handoff.tar.gz
    SRC="$(find rag_handoff -name drafts.jsonl | head -1)"
  fi
fi
[ -f "$SRC" ] || { echo "[오류] drafts.jsonl 을 찾지 못함"; exit 1; }
echo "   drafts: $SRC"

echo "== 2. 위험도·신호등·주요 요인을 최종 모델 값으로 교체"
if [ -f "$REFRESH" ]; then
  rm -rf rag_handoff_refreshed
  python "$WIN/refresh_risk_in_drafts.py" apply --drafts "$SRC" --refresh "$REFRESH" --out-dir rag_handoff_refreshed
  DRAFTS="$WORK/rag_handoff_refreshed/drafts.jsonl"
else
  echo "   [주의] $REFRESH 가 없어 이전 위험도 그대로 진행합니다"
  DRAFTS="$(readlink -f "$SRC")"
fi

echo "== 3. RAG 검증 실행 (${LLM_REPO:-Qwen/Qwen3.5-9B} 기본 모델, 4비트)"
export DRAFTS_PATH="$DRAFTS"
export PHARM_DIR="${PHARM_DIR:-$DL}"
export PHARM_GLOB="${PHARM_GLOB:-Pharmacy_DB-20261013*.zip}"
[ -n "$LLM_PATH" ] && export LLM_PATH      # 모델 폴더를 직접 지정할 때만 (기본은 Hugging Face 캐시에서 찾음)
export LLM_REPO="${LLM_REPO:-Qwen/Qwen3.5-9B}"
[ -n "$GOLD_PATH" ] && export GOLD_PATH
[ -n "$TUNE_CASES_PATH" ] && export TUNE_CASES_PATH
cp "$WIN/rag_verifier_pc.ipynb" .
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 rag_verifier_pc.ipynb --output rag_verifier_pc_executed.ipynb

echo "== 4. 결과를 Windows 폴더로 복사"
DEST="$WIN/rag_verifier_out_$(date +%m%d_%H%M)${MAX_CASES:+_smoke$MAX_CASES}${RUN_TAG:+_$RUN_TAG}"
mkdir -p "$DEST"
cp -r rag_verifier_out/. "$DEST/"
cp rag_verifier_pc_executed.ipynb "$DEST/"
rm -f "$DEST/llm_cache.jsonl"
echo "완료: $DEST"
echo "  rag_result_template.csv (채점·전달용), case_summary.csv, dashboard/ hold/ eval/"
