#!/usr/bin/env bash
# 약물 플래그 수정본으로 위험 모델 쪽 결과를 한 번에 다시 만든다 (CPU 작업, 처방 LLM 학습·평가는 건드리지 않음)
#
# 하는 일 (순서대로, 끝난 단계는 다시 돌려도 안전)
#   0) 점검: 환경·폴더·디스크, xgboost 설치, 지금 결과 파일을 *_before_fix 로 백업 (한 번만)
#   1) patch   NSAID·ACEi/ARB 플래그를 성분명 기준으로 고친 복사본을 만든다
#                - 48시간: _work/team_fixed/            (원본 _work/team 은 그대로)
#                - 이동 시점 24~72h: ~/aki_anchor/T*/data/features/wide.parquet 을 고치고, 원본은 wide_before_fix.parquet 로 보관
#   2) trend   수정본으로 48시간 단일 시점 모델(+추세) 학습·저장  -> _work/team_fixed/risk_scores_trend.parquet, results/eval_team_risk_trend.json
#   3) dynamic 수정본으로 이동 시점 모델 학습·평가·저장 (신호등 기준은 시점마다)  -> results/eval_team_risk_dynamic.json
#   4) models  LightGBM / XGBoost / 랜덤포레스트 / 로지스틱 회귀 비교 (시드 3개)  -> results/eval_team_risk_models.json
#   5) druglist 의료진 확인용 약물군 목록 CSV (성분명·건수)  -> results/drug_list_for_clinicians.csv
#   6) summary 핵심 숫자 요약  -> results/summary_fixed.txt
#
# 처방 LLM(9B)에 쓰인 위험 점수 파일(_work/team/risk_scores_trend.parquet, data_team_trend)은 건드리지 않는다.
#
# 사용 (WSL, 창은 닫지 않기):
#   nohup bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_run_all_fix.sh" > ~/run_all_fix.log 2>&1 &
#   tail -f ~/run_all_fix.log          (끝나면 Ctrl+C 로 로그 보기만 멈춤)
# 옵션 (환경 변수):
#   STEPS="patch trend dynamic models druglist summary"   돌릴 단계 (기본 전부)
#   ANCHORS="24 30 36 42 54 60 66 72"                     이동 시점 (48 은 _work/team_fixed 를 쓰므로 넣지 않음)
#   THREADS=4                                             모델 비교 스레드 수
set -uo pipefail

WIN_DIR="$(cd "$(dirname "$0")" && pwd)"
AKI_VENV="${AKI_VENV:-$HOME/aki_venv}"       # duckdb 가 있는 환경 (팀원 전처리)
KCD_VENV="${KCD_VENV:-$HOME/kcd_venv}"       # lightgbm 등이 있는 환경
AKI_PROJ="${AKI_PROJ:-$HOME/aki_pipeline}"
ROOT="${ANCHOR_ROOT:-$HOME/aki_anchor}"
ANCHORS="${ANCHORS:-24 30 36 42 54 60 66 72}"
STEPS="${STEPS:-patch trend dynamic models druglist summary}"
THREADS="${THREADS:-4}"
PYA="$AKI_VENV/bin/python"
PYK="$KCD_VENV/bin/python"
t_start=$(date +%s)

say() { echo; echo "== [$(date '+%m-%d %H:%M')] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
fail=0
bad() { echo "[주의] $*"; fail=1; }

cd "$WIN_DIR" || exit 1
export PYTHONIOENCODING=utf-8
unset TRITON_OVERRIDE_ARCH PYTORCH_CUDA_ALLOC_CONF

# ---------------------------------------------------------------- 0. 점검
say "0/6 점검"
[ -x "$PYA" ] || { echo "[오류] $PYA 가 없습니다 (팀원 전처리 환경)"; exit 1; }
[ -x "$PYK" ] || { echo "[오류] $PYK 가 없습니다 (kcd_venv)"; exit 1; }
for f in train_risk_trend.py train_risk_anchors.py compare_models.py patch_drug_flags.py export_drug_list.py; do
  [ -f "$f" ] || { echo "[오류] $WIN_DIR/$f 가 없습니다. 최신 파일을 폴더에 넣어 주세요."; exit 1; }
done
for f in wide.parquet cohort_final.parquet splits.parquet default_features.json trend_features.parquet; do
  [ -f "_work/team/$f" ] || { echo "[오류] _work/team/$f 가 없습니다"; exit 1; }
done
[ -f "$AKI_PROJ/data/mimiciv/3.1_parquet/hosp/prescriptions.parquet" ] || { echo "[오류] 처방 표가 없습니다: $AKI_PROJ/data/mimiciv/3.1_parquet/hosp/prescriptions.parquet"; exit 1; }
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 5 ] || { echo "[오류] 디스크 여유 ${FREE_GB}GB (5GB 이상 필요)"; exit 1; }
if pgrep -f "train_qlora.py|eval_sample.py" >/dev/null 2>&1; then
  echo "[참고] 처방 LLM 학습·평가가 도는 중입니다. 이 작업은 CPU 만 쓰지만 메모리를 나눠 쓰므로 느려질 수 있습니다."
fi
"$PYK" -c "import lightgbm, sklearn, pandas" 2>/dev/null || { echo "[오류] kcd_venv 에 lightgbm/scikit-learn/pandas 가 없습니다"; exit 1; }
"$PYA" -c "import duckdb, pandas" 2>/dev/null || { echo "[오류] aki_venv 에 duckdb/pandas 가 없습니다"; exit 1; }
if has models && ! "$PYK" -c "import xgboost" 2>/dev/null; then
  echo "xgboost 설치 시도"; "$PYK" -m pip install -q xgboost || bad "xgboost 설치 실패 -> 모델 비교에서 XGBoost 는 빠집니다"
fi
# 백업: 한 번만 (이미 있으면 건너뜀)
bk() { [ -e "$1" ] && [ ! -e "$2" ] && cp -r "$1" "$2" && echo "백업: $2"; true; }
bk results/eval_team_risk_dynamic.json results/eval_team_risk_dynamic_before_fix.json
bk results/eval_team_risk_trend.json   results/eval_team_risk_trend_after_flag_fix_first_run.json
bk runs/aki_risk_dynamic               runs/aki_risk_dynamic_before_fix
bk runs/aki_risk_team_trend            runs/aki_risk_team_trend_before_fix
bk _work/anchors/risk_scores_dynamic.parquet _work/anchors/risk_scores_dynamic_before_fix.parquet
echo "단계: $STEPS | 이동 시점: $ANCHORS | 스레드 $THREADS"

# ---------------------------------------------------------------- 1. 약물 플래그 수정
if has patch; then
  say "1/6 약물 플래그 수정 (NSAID·ACEi/ARB 성분명 기준)"
  nice -n 10 "$PYA" patch_drug_flags.py --proj "$AKI_PROJ" --team-dir _work/team --out _work/team_fixed \
    || { echo "[오류] 48시간 수정 실패 - 중단"; exit 1; }
  for T in $ANCHORS; do
    D="$ROOT/T$T/data/features"
    if [ ! -f "$D/wide.parquet" ]; then bad "T=${T}h 자료가 없어 건너뜀: $D (wsl_multi_anchor.sh 필요)"; continue; fi
    echo "-- T=${T}h"
    nice -n 10 "$PYA" patch_drug_flags.py --proj "$AKI_PROJ" --team-dir "$D" --inplace 2>&1 | sed -n '1,6p' \
      || bad "T=${T}h 수정 실패"
  done
fi
[ -f _work/team_fixed/wide.parquet ] || { echo "[오류] _work/team_fixed/wide.parquet 이 없습니다 (patch 단계 필요)"; exit 1; }

# ---------------------------------------------------------------- 2. 48시간 단일 시점 모델
if has trend; then
  say "2/6 48시간 단일 시점 모델(+추세) - 수정본으로 학습·저장"
  nice -n 10 "$PYK" train_risk_trend.py --team-dir _work/team_fixed --save 2>&1 | grep -vE "Warning|eval_set" \
    || bad "trend 단계 실패"
fi

# ---------------------------------------------------------------- 3. 이동 시점 모델
if has dynamic; then
  say "3/6 이동 시점 모델 - 수정본, 신호등 기준은 시점마다"
  nice -n 10 "$PYK" train_risk_anchors.py --team-dir _work/team_fixed --anchors-root "$ROOT" --save --tier-mode per_anchor 2>&1 | grep -vE "Warning|eval_set|df\[" \
    || bad "dynamic 단계 실패"
fi

# ---------------------------------------------------------------- 4. 알고리즘 비교
if has models; then
  say "4/6 알고리즘 비교 (LightGBM / XGBoost / 랜덤포레스트 / 로지스틱)"
  nice -n 10 "$PYK" compare_models.py --team-dir _work/team_fixed --threads "$THREADS" 2>&1 | grep -vE "Warning|eval_set" \
    || bad "models 단계 실패"
fi

# ---------------------------------------------------------------- 5. 약물 목록 CSV
if has druglist; then
  say "5/6 의료진 확인용 약물군 목록 CSV"
  nice -n 10 "$PYA" export_drug_list.py --proj "$AKI_PROJ" --wide _work/team_fixed/wide.parquet --out results/drug_list_for_clinicians.csv 2>&1 | head -14 \
    || bad "druglist 단계 실패"
fi

# ---------------------------------------------------------------- 6. 요약
if has summary; then
  say "6/6 요약 -> results/summary_fixed.txt"
  "$PYK" - > results/summary_fixed.txt <<'PY'
import json, os
def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
f = lambda v: "-" if v is None else "%.3f" % v
pc = lambda v: "-" if v is None else "%.1f%%" % (100 * v)
t = load("results/eval_team_risk_trend.json")
print("[48시간 단일 시점 모델, 수정 플래그]")
if t:
    print("  기준 특징 test AUROC %s -> +추세 %s (AUPRC %s)" % (f(t["base"]["test_lightgbm"]["auroc"]), f(t["trend"]["test_lightgbm"]["auroc"]), f(t["trend"]["test_lightgbm"]["auprc"])))
d = load("results/eval_team_risk_dynamic.json")
print("\n[이동 시점 모델, 신호등 기준 = %s]" % (d["thresholds"].get("tier_mode", "pooled") if d else "-"))
if d:
    print("  시점  사례  양성  AUROC  따로학습 | 빨강 알림 민감도 양성예측도 | 노랑이상 알림 민감도 양성예측도")
    for k, r in d["per_anchor_test"].items():
        a, b = r["red"], r["yellow_or_red"]
        print("  %3sh %5s %5s  %s  %s | %s %s %s | %s %s %s" % (k, r["n"], r.get("positives", "-"), f(r["auroc"]), f(r.get("separate_model_auroc")),
              pc(a["alert_rate"]), pc(a["sensitivity"]), pc(a["ppv"]), pc(b["alert_rate"]), pc(b["sensitivity"]), pc(b["ppv"])))
    c = d.get("compare_at_48h")
    if c:
        print("  48시간 같은 %s건: 이동 시점 AUROC %s / AUPRC %s | 단일 시점 AUROC %s / AUPRC %s | 차이 AUROC %s, AUPRC %s (95%% 구간)" % (
            c["cases"], f(c["dynamic_model"]["auroc"]), f(c["dynamic_model"]["auprc"]), f(c["single_time_trend_model"]["auroc"]),
            f(c["single_time_trend_model"]["auprc"]), c["dynamic_minus_single_bootstrap"]["auroc_diff_ci95"], c["dynamic_minus_single_bootstrap"]["auprc_diff_ci95"]))
    for k, v in d["episode_level_test"].items():
        print("  환자 단위 %-13s AKI %s명 중 발생 전 알림 %s, 여유 시간 중앙값 %s시간, 12시간 이상 앞서 %s | AKI 없던 환자 중 알림 경험 %s" % (
            k, v["aki_stays"], pc(v["aki_stays_alerted_rate"]), v.get("lead_time_hours_median", "-"), pc(v.get("alerted_at_least_12h_before_onset_rate")),
            pc(v["non_aki_stays_ever_alerted_rate"])))
m = load("results/eval_team_risk_models.json")
print("\n[알고리즘 비교, test AUROC (시드 평균 [범위]) / LightGBM 대비 차이 95% 구간]")
if m:
    for k, v in m["models"].items():
        bd = v.get("minus_lightgbm_bootstrap")
        print("  %-18s %s [%s-%s]  AUPRC %s  %s" % (k, f(v["auroc_mean"]), f(v["auroc_min"]), f(v["auroc_max"]), f(v["auprc_mean"]),
              ("차이 %s" % bd["auroc_diff_ci95"]) if bd else "(기준)"))
else:
    print("  (결과 없음)")
PY
  cat results/summary_fixed.txt
fi

say "끝 ($(( ($(date +%s) - t_start) / 60 ))분 걸림)$([ $fail = 1 ] && echo ' - 위에 [주의] 항목이 있습니다')"
