#!/usr/bin/env bash
# WSL 학습 환경 설치 (처음 한 번). 4비트 QLoRA + Qwen3.5 선형 어텐션 가속(flash-linear-attention)
#
# 사용 (WSL 터미널에서):
#   bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_setup.sh"
# 바꿀 수 있는 값: VENV(가상환경 위치, 기본 ~/kcd_venv), TORCH_INDEX(기본 CUDA 12.8 용), WITH_CAUSAL_CONV=0(빌드 생략)
#
# 주의: GPU 드라이버는 Windows 에만 설치합니다. WSL 안에 NVIDIA 드라이버를 깔지 마세요.
set -euo pipefail
WIN_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/kcd_venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "== GPU 확인"
if ! command -v nvidia-smi >/dev/null; then
  echo "[오류] WSL 에서 GPU 가 보이지 않습니다. Windows 의 NVIDIA 드라이버를 최신으로 업데이트한 뒤 wsl --shutdown 후 다시 여세요."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== 기본 패키지 (sudo 비밀번호를 물을 수 있습니다)"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev build-essential rsync git

echo "== 가상환경 $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel setuptools packaging ninja

echo "== PyTorch (RTX 50 시리즈는 CUDA 12.8 이상 빌드 필요)"
pip install torch --index-url "$TORCH_INDEX"

echo "== 학습 라이브러리 (Windows 환경과 같은 버전)"
pip install "transformers==5.19.0" "peft==0.21.2" "bitsandbytes==0.50.2" accelerate pandas pyarrow numpy safetensors

echo "== 선형 어텐션 가속 (핵심)"
pip install flash-linear-attention || echo "[주의] flash-linear-attention 설치 실패 - 느린 기본 구현으로 동작합니다"

if [ "${WITH_CAUSAL_CONV:-1}" = "1" ]; then
  echo "== causal-conv1d (선택, CUDA 툴킷이 있으면 빌드. 실패해도 학습에는 지장 없음)"
  if command -v nvcc >/dev/null; then
    MAX_JOBS=4 pip install --no-build-isolation causal-conv1d || echo "[참고] causal-conv1d 빌드 실패 - 생략합니다"
  else
    echo "[참고] nvcc 가 없어 causal-conv1d 를 생략합니다 (flash-linear-attention 만으로 대부분의 속도 이득을 얻습니다)"
  fi
fi

echo "== Windows 에 받아 둔 Qwen3.5-9B 가 있으면 WSL 캐시로 복사 (다시 내려받지 않도록)"
WIN_HF="/mnt/c/Users/${WIN_USER:-User}/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
WSL_HF="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
if [ -d "$WIN_HF" ] && [ ! -d "$WSL_HF" ]; then
  mkdir -p "$(dirname "$WSL_HF")"
  cp -rL "$WIN_HF" "$WSL_HF" && echo "복사 완료" || echo "[참고] 복사 실패 - 학습 시작 때 새로 내려받습니다 (약 20GB)"
fi

echo "== 점검"
cd "$WIN_DIR"
python check_env.py || true
python - <<'PY'
import torch
print("CUDA 사용:", torch.cuda.is_available(), "| 연산능력:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
try:
    import fla, triton
    print("flash-linear-attention", getattr(fla, "__version__", "?"), "| triton", triton.__version__, "-> 가속 사용")
except Exception as e:
    print("[주의] 가속 라이브러리를 불러오지 못함:", e)
PY
echo "설치 끝. 다음: bash \"$WIN_DIR/wsl_train.sh\""
