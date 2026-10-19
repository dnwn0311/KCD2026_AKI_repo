#!/usr/bin/env bash
# 밤새 한 번에: 추세 특징을 넣은 처방 LLM(9B) 실험 - 전처리 -> 학습 -> 평가 -> 기존 9B 와 비교
#
# 바꾸는 점 (기존 9B 팀원 기준 모델 qlora_team 과 비교)
#   1) 프롬프트의 위험 점수를 추세 특징 위험 모델(AUROC 0.769) 값으로 바꾼다
#   2) 프롬프트에 [최근 변화] 줄을 더한다 (크레아티닌 24시간 변화, 소변 저하 시간, 혈압 최저, 수분 균형, 승압제)
#   정답(처방 변경)과 사례는 그대로라 같은 test 600건에서 바로 비교할 수 있다
#
# 단계 (이미 끝난 단계는 건너뛰므로, 중간에 멈추면 같은 명령을 다시 실행하면 이어서 한다)
#   1) 추세 위험 모델 저장           -> _work/team/risk_scores_trend.parquet                (수 분)
#   2) 처방 데이터 만들기            -> data_team_trend/{train,val,test}.jsonl               (10~30분)
#   3) 9B QLoRA 학습                 -> runs/qlora_team_trend/adapter                        (가장 오래 걸림)
#   4) 평가: val 200건으로 τ, test 600건 -> results/eval_team9trend_test.json
#   5) 비교용: 기존 9B(qlora_team)의 test 600건 -> results/eval_team9_test.json
#   6) 요약                           -> results/night_trend_summary.txt
#
# 사용 (WSL 터미널, 창은 닫지 않기):
#   nohup bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_night_trend.sh" > ~/night_trend.log 2>&1 &
# 진행 확인:  tail -5 ~/night_trend.log
set -uo pipefail
WIN_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/kcd_rx}"
VENV="${VENV:-$HOME/kcd_venv}"
RUN=qlora_team_trend
DATA=data_team_trend
SUMMARY="$WIN_DIR/results/night_trend_summary.txt"

t_start=$(date +%s)
say() { echo; echo "== [$(date '+%m-%d %H:%M')] $*"; }
elapsed() { echo "$(( ($(date +%s) - t_start) / 60 ))분 경과"; }

[ -f "$VENV/bin/activate" ] || { echo "[오류] $VENV 가 없습니다"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if pgrep -f "train_qlora.py|eval_sample.py" >/dev/null; then
  echo "[중단] 다른 학습·평가가 GPU 를 쓰고 있습니다. 끝난 뒤 다시 실행하세요."; exit 1
fi
python -c "import lightgbm, torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))" \
  || { echo "[오류] lightgbm 또는 GPU 확인 실패 (pip install lightgbm scikit-learn)"; exit 1; }

# ---- 1. 추세 위험 모델 저장
say "1/6 추세 위험 모델 저장"
cd "$WIN_DIR"
mkdir -p _work/team
for f in data/cohort/cohort_final.parquet data/features/wide.parquet data/features/splits.parquet data/features/default_features.json; do
  [ -f "_work/team/$(basename "$f")" ] || cp "$HOME/aki_pipeline/$f" _work/team/ || { echo "[오류] 팀원 전처리 파일 $f 없음"; exit 1; }
done
if [ -f "_work/team/risk_scores_trend.parquet" ]; then
  echo "이미 있음: _work/team/risk_scores_trend.parquet"
else
  python train_risk_trend.py --save || { echo "[오류] 1단계 실패"; exit 1; }
fi

# ---- 2. 처방 데이터 (위험 점수 교체 + [최근 변화] 줄)
say "2/6 처방 데이터 만들기 ($DATA)"
if [ -f "$DATA/test.jsonl" ]; then
  echo "이미 있음: $DATA/"
else
  python prep_rx_team.py --risk-scores risk_scores_trend.parquet --trend-line --out-dir "$WIN_DIR/$DATA" \
    || { echo "[오류] 2단계 실패"; exit 1; }
  python - <<'PY'
import json
r = json.loads(open("data_team_trend/test.jsonl", encoding="utf-8").readline())
print("[예시 프롬프트 마지막 4줄]")
print("\n".join(r["messages"][1]["content"].splitlines()[-4:]))
PY
fi
elapsed

# ---- 3. 학습 (기존 wsl_train_team.sh 그대로: 자동 재시작, 이어서 학습)
say "3/6 9B QLoRA 학습 (runs/$RUN)"
RUN="$RUN" DATA_DIR="$DATA" bash "$WIN_DIR/wsl_train_team.sh" || { echo "[오류] 3단계 실패"; exit 1; }
elapsed

# ---- 4. 평가 (추세 모델)
cd "$WORK"
# 평가는 expandable_segments 없이 실행 (WSL 에서 'CUDA driver error: device not ready' 가 난 적이 있음; 학습은 wsl_train_team.sh 가 자체 설정)
unset PYTORCH_CUDA_ALLOC_CONF
say "4/6 평가: val 200건으로 τ 찾기 -> test 600건"
export KCD_DATA_DIR="$DATA"
if [ ! -f results/tau_team9trend_val.json ]; then
  python eval_sample.py --adapter "runs/$RUN/adapter" --split val --limit 200 --n 16 --tag team9trend_val \
    || { echo "[오류] 4단계(val) 실패"; exit 1; }
fi
if [ ! -f results/eval_team9trend_test.json ]; then
  python eval_sample.py --adapter "runs/$RUN/adapter" --split test --limit 600 --n 16 --tag team9trend_test \
    --tau results/tau_team9trend_val.json || { echo "[오류] 4단계(test) 실패"; exit 1; }
fi
elapsed

# ---- 5. 비교용: 기존 9B 팀원 기준 모델의 test (같은 600건)
say "5/6 비교용: 기존 9B(qlora_team) test 600건"
export KCD_DATA_DIR=data_team
if [ -f results/eval_team9_test.json ]; then
  echo "이미 있음"
elif [ -d runs/qlora_team/adapter ] && [ -f results/tau_team_val.json ]; then
  python eval_sample.py --adapter runs/qlora_team/adapter --split test --limit 600 --n 16 --tag team9_test \
    --tau results/tau_team_val.json || echo "[주의] 5단계 실패 (추세 모델 결과는 이미 저장됨)"
else
  echo "[건너뜀] runs/qlora_team/adapter 또는 results/tau_team_val.json 이 없습니다"
fi
elapsed

# ---- 6. 요약
say "6/6 요약 -> $SUMMARY"
python - > "$SUMMARY" <<'PY'
import json, os
ACT = ["신규처방", "감량", "증량", "중단"]
def row(name, path, key=None):
    if not os.path.exists(path):
        return "  %-34s (없음)" % name
    d = json.load(open(path, encoding="utf-8"))
    o = d[key]["overall"] if key else d["overall"]
    ba = o.get("by_action", {})
    f = lambda v: "  -  " if v is None else "%.3f" % v
    return "  %-34s 전체 %s | 신규 %s 감량 %s 증량 %s 중단 %s" % (
        name, f(o.get("micro_f1")), *[f((ba.get(k) or {}).get("f1")) for k in ACT])
print("처방 변경 초안 test 600건 비교 (F1, val 에서 정한 τ 적용)")
print(row("빈도 기준선 (전체 test)", "results/baseline_data_team.json", "freq_rule"))
print(row("기존 9B (qlora_team)", "results/eval_team9_test.json"))
print(row("추세 9B (qlora_team_trend)", "results/eval_team9trend_test.json"))
print(row("27B (qlora_team27, 서버)", "results/eval_team27_test.json"))
r = "results/eval_team_risk_trend.json"
if os.path.exists(r):
    d = json.load(open(r, encoding="utf-8"))
    print("\n위험 모델 test AUROC: 기존 %.3f -> 추세 %.3f" % (d["base"]["test_lightgbm"]["auroc"], d["trend"]["test_lightgbm"]["auroc"]))
PY
cat "$SUMMARY"
say "완료 ($(elapsed))"
