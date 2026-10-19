#!/usr/bin/env bash
# 이동 시점(여러 예측 시점) 전처리: 팀원 파이프라인을 시점(time_zero)마다 따로 돌려 시점별 코호트·특징·정답을 만든다
#
# 시점 T 마다 하는 일
#   - 팀원 코드(~/aki_pipeline)를 ~/aki_anchor/T<T>/ 로 복사하고 config.yaml 의 두 값만 바꾼다
#       task.time_zero_hours  = T      (입실 후 T시간에 예측)
#       cohort.min_icu_los_hours = T   (그 시점에 ICU 에 있는 환자만)
#     나머지 규칙(AKI 정의, 이미 AKI 가 생긴 환자 제외, 정답 = T 이후 24시간 안 첫 AKI, 데이터 누수 방지)은 그대로
#   - 02 03 04 05 (코호트·정답) -> 06 (약물 매핑, 캐시 사용) -> 07 (시간별·요약 특징) 실행
#   - 01(CSV 변환)은 다시 하지 않고 ~/aki_pipeline 의 변환 결과를 연결해서 쓴다
#   48시간 시점은 기존 결과(~/aki_pipeline)를 그대로 쓰므로 다시 만들지 않는다
#
# 한 시점에 몇 분 걸린다 (04 약 1~2분, 07 1분 미만, 02·03·05 는 짧음). 중간에 멈춰도 같은 명령을 다시 실행하면
# 끝난 시점은 건너뛰고 이어서 한다.
#
# 사용 (WSL, 창은 닫지 않기):
#   nohup bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_multi_anchor.sh" > ~/multi_anchor.log 2>&1 &
#   tail -f ~/multi_anchor.log
# 옵션 (환경 변수):
#   ANCHORS="24 30 36 42 54 60 66 72"   만들 시점 (기본값). 48 은 넣지 않는다
#   AKI_PROJ=~/aki_pipeline  AKI_VENV=~/aki_venv  ANCHOR_ROOT=~/aki_anchor
#   FORCE=1   처방 모델 학습이 도는 중에도 실행 (권하지 않음)
#
# 환자 단위 자료는 이 PC 안에서만 만들어진다. 어디에도 올리지 않는다.
set -uo pipefail

PROJ="${AKI_PROJ:-$HOME/aki_pipeline}"
VENV="${AKI_VENV:-$HOME/aki_venv}"
ROOT="${ANCHOR_ROOT:-$HOME/aki_anchor}"
ANCHORS="${ANCHORS:-24 30 36 42 54 60 66 72}"
PY="$VENV/bin/python"
STEPS="02_build_base_cohort 03_add_esrd_flag 04_build_labels 05_build_final_cohort 06_build_drug_map 07_build_features"

say() { echo; echo "== [$(date '+%m-%d %H:%M')] $*"; }

[ -x "$PY" ] || { echo "[오류] $PY 가 없습니다. 먼저 wsl_team_pipeline.sh 를 실행했는지 확인하세요."; exit 1; }
[ -d "$PROJ/scripts" ] && [ -f "$PROJ/config.yaml" ] || { echo "[오류] $PROJ 에 팀원 파이프라인(scripts, config.yaml)이 없습니다."; exit 1; }
[ -e "$PROJ/data/mimiciv/3.1_parquet" ] || { echo "[오류] 변환된 MIMIC 표가 없습니다: $PROJ/data/mimiciv/3.1_parquet"; exit 1; }
[ -f "$PROJ/data/features/hourly.parquet" ] || echo "[참고] 48시간 기준 결과(hourly.parquet)가 $PROJ 에 없습니다. 학습 단계에서 필요합니다."
if pgrep -f "train_qlora.py" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  echo "[중단] 처방 모델 학습이 도는 중입니다 (메모리 부족으로 죽을 수 있음). 끝난 뒤 다시 실행하거나 FORCE=1."; exit 1
fi
if pgrep -f "eval_sample.py" >/dev/null 2>&1; then
  echo "[참고] 평가(eval_sample.py)가 도는 중입니다. 같이 돌려도 되지만 메모리를 나눠 쓰므로 조금 느려질 수 있습니다."
fi
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 15 ] || { echo "[오류] 디스크 여유 ${FREE_GB}GB (15GB 이상 필요)"; exit 1; }
mkdir -p "$ROOT"
if curl -sS -m 15 -o /dev/null "https://rxnav.nlm.nih.gov/REST/version.json" 2>/dev/null; then NET=1; else NET=0; fi
echo "시점: $ANCHORS | 작업 폴더: $ROOT | 인터넷(RxNav): $([ $NET = 1 ] && echo 가능 || echo 불가 -> 06단계는 48시간용 약물 매핑을 복사해 씀)"

for T in $ANCHORS; do
  D="$ROOT/T$T"
  if [ -f "$D/data/features/wide.parquet" ] && [ -f "$D/data/features/hourly.parquet" ] && [ -f "$D/data/cohort/cohort_final.parquet" ]; then
    echo; echo "== T=${T}h 이미 있음: $D (건너뜀)"; continue
  fi
  say "T=${T}h 준비"
  mkdir -p "$D/data/mimiciv" "$D/data/external"
  rm -rf "$D/scripts"; cp -r "$PROJ/scripts" "$D/scripts"
  cp "$PROJ/config.yaml" "$D/config.yaml"
  "$PY" - "$D/config.yaml" "$T" <<'PYEOF'
import sys, yaml
p, T = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(p, encoding="utf-8"))
c["task"]["time_zero_hours"] = T
c["cohort"]["min_icu_los_hours"] = T
yaml.safe_dump(c, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("config: time_zero_hours=%d, min_icu_los_hours=%d, horizon_hours=%s" % (T, T, c["task"]["horizon_hours"]))
PYEOF
  ln -sfn "$PROJ/data/mimiciv/3.1_parquet" "$D/data/mimiciv/3.1_parquet"
  [ -f "$PROJ/data/external/rxnav_cache.json" ] && cp "$PROJ/data/external/rxnav_cache.json" "$D/data/external/"
  cd "$D" || exit 1
  ok=1
  for s in $STEPS; do
    n="${s%%_*}"
    if [ "$n" = "06" ] && [ "$NET" = "0" ]; then
      cp "$PROJ/data/external/drug_map.parquet" "$PROJ/data/external/drug_map_info.json" data/external/ 2>/dev/null \
        && echo "-- 06 건너뜀: 48시간용 drug_map.parquet 복사 (새로 생긴 약은 매핑 없이 빈 값)" || { echo "[오류] drug_map.parquet 이 없습니다"; ok=0; break; }
      continue
    fi
    say "T=${T}h  $s"
    "$PY" -I "scripts/$s.py" || { echo "[오류] T=${T}h $s 실패"; ok=0; break; }
  done
  if [ "$ok" = "1" ]; then
    "$PY" - <<'PYEOF'
import duckdb
r = duckdb.sql("select count(*) n, sum(label_aki_next_24h::int) pos, round(avg(label_aki_next_24h::int),4) rate from 'data/features/wide.parquet'").fetchone()
print("[결과] 사례 %d건, 양성 %d건 (%.1f%%)" % (r[0], r[1], 100 * r[2]))
PYEOF
  else
    echo "[주의] T=${T}h 는 실패했습니다. 위 오류를 확인한 뒤 같은 명령을 다시 실행하세요 (끝난 시점은 건너뜁니다)"
  fi
  cd "$HOME" || true
done

say "시점별 요약"
"$PY" - "$ROOT" "$PROJ" $ANCHORS <<'PYEOF'
import os, sys, duckdb
root, proj, anchors = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
rows = []
for T in sorted(set(anchors + [48])):
    d = proj if T == 48 else os.path.join(root, "T%d" % T)
    w = os.path.join(d, "data/features/wide.parquet")
    if not os.path.exists(w):
        rows.append((T, None, None, None)); continue
    n, pos = duckdb.sql("select count(*), sum(label_aki_next_24h::int) from '%s'" % w).fetchone()
    rows.append((T, n, pos, pos / n))
print("시점(시간)  사례   양성   양성률")
for T, n, pos, r in rows:
    print("%8s  %6s %6s %6s" % (T, n if n is not None else "-", pos if pos is not None else "-", "%.1f%%" % (100 * r) if r is not None else "-"))
print("\n다음:  python train_risk_anchors.py --smoke   (점검)  ->  python train_risk_anchors.py   (학습·평가)")
PYEOF
