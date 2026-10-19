#!/usr/bin/env bash
# 9B(qlora_team) 로 test 600건 처방 초안을 PC 에서 다시 만들고, 채점용 정답(gold)도 함께 만든다 (WSL).
#   실행:  bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_make_drafts9.sh"
# 하는 일
#   1) 샘플링: test 600건 x 16번 (eval_sample.py, 어댑터 runs/qlora_team, τ=results/tau_team_val.json)
#      -> _work/votes_team9_test_s0.jsonl   (이미 있으면 건너뜀)
#   2) 초안 묶음: make_drafts.py -> _work/rag_handoff_team9_test/drafts.jsonl, _work/rag_labels_team9_test.jsonl(정답)
#   3) 정답을 조정 200 / 평가 400 으로 나눔 (seed 42) -> _work/gold_split/
# 환경 변수: VENV (기본 ~/kcd_venv)   TAG (기본 team9_test)
set -e
WIN="/mnt/c/Users/User/Documents/KCD_처방추천"
VENV="${VENV:-$HOME/kcd_venv}"
TAG="${TAG:-team9_test}"
ADAPTER="${ADAPTER:-runs/qlora_team/adapter}"
TAU="${TAU:-results/tau_team_val.json}"
[ -f "$VENV/bin/activate" ] || { echo "[오류] $VENV 가 없습니다. VENV=... 로 지정하세요."; exit 1; }
if pgrep -f "train_qlora.py|eval_sample.py|nbconvert" >/dev/null; then
  echo "[중단] 다른 학습·평가·RAG 실행이 GPU 를 쓰고 있습니다. 끝난 뒤 다시 실행하세요."; exit 1
fi
source "$VENV/bin/activate"
export PYTHONIOENCODING=utf-8
unset PYTORCH_CUDA_ALLOC_CONF          # 평가는 expandable_segments 없이 (WSL 에서 device not ready 가 난 적이 있음)
if [ -n "${TRITON_OVERRIDE_ARCH:-}" ] && ! [[ "$TRITON_OVERRIDE_ARCH" =~ ^sm[0-9]+$ ]]; then
  echo "   [참고] TRITON_OVERRIDE_ARCH=$TRITON_OVERRIDE_ARCH 는 형식이 맞지 않아 해제합니다"; unset TRITON_OVERRIDE_ARCH
fi
export KCD_DATA_DIR=data_team          # 팀원 기준 데이터
cd "$WIN"
[ -d "$ADAPTER" ] || { echo "[오류] 어댑터 없음: $ADAPTER"; exit 1; }
[ -f "$TAU" ] || { echo "[오류] τ 파일 없음: $TAU"; exit 1; }

echo "== 1/3 샘플링 (test 600건 x 16) - 진행 줄에 남은 시간이 표시됩니다"
if [ -f "_work/votes_${TAG}_s0.jsonl" ] && [ -f "results/eval_${TAG}_s0.json" ]; then
  echo "   이미 있음: _work/votes_${TAG}_s0.jsonl"
else
  python eval_sample.py --adapter "$ADAPTER" --split test --limit 600 --n 16 --tag "${TAG}_s0" --tau "$TAU"
fi

echo "== 2/3 초안 묶음 만들기"
python make_drafts.py --tag "$TAG" --shards 1 --tau "$TAU"
OUT="_work/rag_handoff_${TAG}"
GOLD="_work/rag_labels_${TAG}.jsonl"
[ -f "$OUT/drafts.jsonl" ] && [ -f "$GOLD" ] || { echo "[오류] 초안 또는 정답 파일이 만들어지지 않았습니다"; exit 1; }

echo "== 3/3 정답을 조정 200 / 평가 400 으로 나누기 (seed 42)"
python - "$GOLD" "_work/gold_split" <<'PY'
import json, os, random, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
random.Random(42).shuffle(rows)
tune, hold = rows[:200], rows[200:]
w = lambda p, rs: open(p, "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rs) + "\n")
w(os.path.join(out, "gold_tune200.jsonl"), tune)
w(os.path.join(out, "gold_hold400.jsonl"), hold)
open(os.path.join(out, "tune_ids.txt"), "w").write("\n".join(r["case_id"] for r in tune) + "\n")
open(os.path.join(out, "hold_ids.txt"), "w").write("\n".join(r["case_id"] for r in hold) + "\n")
print("   전체 %d건 -> 조정용 %d / 평가용 %d" % (len(rows), len(tune), len(hold)))
PY

echo
echo "완료. 다음 단계 (조정용 200건 RAG 검증):"
echo "  DRAFTS_IN=\"$WIN/$OUT/drafts.jsonl\" \\"
echo "  CASE_IDS_PATH=\"$WIN/_work/gold_split/tune_ids.txt\" GOLD_PATH=\"$WIN/_work/gold_split/gold_tune200.jsonl\" \\"
echo "  EVAL_SUBSET_N=0 RUN_SYNTHESIS=0 RUN_TAG=tune200 bash \"$WIN/wsl_run_rag.sh\""
