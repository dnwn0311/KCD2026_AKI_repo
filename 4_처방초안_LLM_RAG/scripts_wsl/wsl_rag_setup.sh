#!/usr/bin/env bash
# RAG 검증용 환경 점검 (WSL, 처음 한 번). 처방 모델을 학습·평가한 환경(~/kcd_venv)을 그대로 쓰고, 빠진 패키지만 더한다.
# 환자 정보는 쓰지 않고 공개 패키지·공개 모델만 내려받는다.
#   bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_rag_setup.sh"
# 환경 변수: VENV(기본 ~/kcd_venv)  LLM_REPO(기본 Qwen/Qwen3.5-9B)
set -e
VENV="${VENV:-$HOME/kcd_venv}"
REPO="${LLM_REPO:-Qwen/Qwen3.5-9B}"
[ -f "$VENV/bin/activate" ] || { echo "[오류] $VENV 가 없음. 처방 모델을 돌리던 환경 이름을 VENV=... 로 지정하세요."; exit 1; }
source "$VENV/bin/activate"

echo "== 1. 빠진 패키지만 추가 (torch·transformers 는 건드리지 않음)"
pip install -q ijson openpyxl scipy scikit-learn pandas numpy jupyter nbconvert ipykernel

echo "== 2. GPU·패키지 확인"
REPO="$REPO" python - <<'PY'
import os, glob
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| 사용 가능", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("[오류] GPU 를 못 찾음. WSL 에서 nvidia-smi 가 동작하는지 확인하세요.")
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name, "| VRAM %.1f GB" % (p.total_memory / 1e9))
print("transformers", transformers.__version__)
if int(transformers.__version__.split(".")[0]) < 5 and "3.5" in os.environ["REPO"]:
    print("[주의] Qwen3.5 는 transformers 5.x 가 필요합니다. 처방 모델을 돌릴 때 쓴 환경이 맞는지 확인하세요.")
import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__)
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots/*" % os.environ["REPO"].replace("/", "--")))
print("기본 모델 캐시:", snap[0] if snap else "없음 (처음 실행 때 노트북이 내려받음)")
PY
echo "== 준비 끝. 다음: MAX_CASES=10 bash \"/mnt/c/Users/User/Documents/KCD_처방추천/wsl_run_rag.sh\""
