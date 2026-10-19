#!/usr/bin/env bash
# 서버에서 받은 정답(gold) 암호화 파일을 풀고, 조정 200 / 평가 400 으로 나눈다 (WSL).
#   실행:  bash "/mnt/c/Users/User/Documents/KCD_처방추천/wsl_prep_gold.sh"
# 찾는 곳: _work/ 와 Downloads/ 의 rag_labels*.gpg  (다른 위치면  GPG_GOLD=<파일> 로 지정)
# 결과:    _work/rag_labels_team27_test.jsonl (600줄), _work/gold_split/{gold_tune200,gold_hold400}.jsonl, tune_ids.txt, hold_ids.txt
set -e
WIN="/mnt/c/Users/User/Documents/KCD_처방추천"
DL="/mnt/c/Users/User/Downloads"
OUT="$WIN/_work/rag_labels_team27_test.jsonl"
SPLIT="$WIN/_work/gold_split"

if [ -n "$GPG_GOLD" ]; then
  SRC="$GPG_GOLD"
else
  SRC="$(ls -t "$WIN"/_work/rag_labels*.gpg "$DL"/rag_labels*.gpg 2>/dev/null | head -1)"
fi
[ -n "$SRC" ] && [ -f "$SRC" ] || { echo "[오류] rag_labels*.gpg 를 _work 또는 Downloads 에서 찾지 못했습니다. GPG_GOLD=<파일 경로> 로 지정하세요."; exit 1; }
echo "정답 암호화 파일: $SRC"

if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -eq 600 ]; then
  echo "이미 풀려 있음 (600줄): $OUT"
else
  export GPG_TTY="$(tty)"
  echo "암호를 입력하세요 (화면에 표시되지 않음)"
  gpg --pinentry-mode loopback -o "$OUT.tmp" -d "$SRC"
  mv "$OUT.tmp" "$OUT"
fi

N="$(wc -l < "$OUT")"
[ "$N" -eq 600 ] || { echo "[오류] 정답이 $N 줄입니다 (600줄이어야 함). 암호가 틀렸거나 다른 파일일 수 있습니다."; rm -f "$OUT"; exit 1; }

python3 - "$OUT" "$SPLIT" <<'PY'
import json, os, random, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
ids = [r["case_id"] for r in rows]
assert len(rows) == 600 and len(set(ids)) == 600, (len(rows), len(set(ids)))
random.Random(42).shuffle(rows)
tune, hold = rows[:200], rows[200:]
w = lambda p, rs: open(p, "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rs) + "\n")
w(os.path.join(out, "gold_tune200.jsonl"), tune)
w(os.path.join(out, "gold_hold400.jsonl"), hold)
open(os.path.join(out, "tune_ids.txt"), "w").write("\n".join(r["case_id"] for r in tune) + "\n")
open(os.path.join(out, "hold_ids.txt"), "w").write("\n".join(r["case_id"] for r in hold) + "\n")
print("정답 600건 -> 조정용 %d / 평가용 %d (seed 42)  ->  %s" % (len(tune), len(hold), out))
PY

echo
echo "다음: 조정용 200건 RAG 검증 (샘플링이 돌고 있으면 먼저 Ctrl+C 로 멈추세요)"
echo "  CASE_IDS_PATH=\"$SPLIT/tune_ids.txt\" GOLD_PATH=\"$SPLIT/gold_tune200.jsonl\" \\"
echo "  EVAL_SUBSET_N=0 RUN_SYNTHESIS=0 RUN_TAG=tune200 bash \"$WIN/wsl_run_rag.sh\""
