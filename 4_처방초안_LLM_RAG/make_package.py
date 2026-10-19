# -*- coding: utf-8 -*-
"""
팀원 전달 패키지 만들기

모으는 것 (환자 단위 자료는 넣지 않음)
  runs/qlora_aki/adapter/   처방 LLM 의 LoRA 어댑터 (기본 모델 Qwen3.5-9B 전용)
  runs/aki_risk/            위험 예측 모델 (model.txt, config.json: 특징 순서·보정·신호등 기준)
  *.py, requirements.txt    같은 입력을 만들기 위한 코드와 라이브러리 버전
  run_받는쪽.sh / .bat       받는 쪽 실행 순서 (전처리 → 받은 모델로 점수 → LLM 입력 → 초안 생성)
  README_받는분.md          사용 안내
  info/                     라이브러리 버전, 파일 체크섬, 보낸 쪽 성능(집계) - 재현 확인용

넣지 않는 것: data/, _work/, results 의 환자 단위 파일, 학습 체크포인트, .venv

출력: 전달패키지_<날짜>/ 폴더와 같은 이름의 .zip
사용:  python make_package.py [--adapter runs/qlora_aki/adapter] [--no-zip]
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from importlib import metadata

import common as C

CODE = ["common.py", "llm_common.py", "prep_extract.py", "prep_build.py", "prep_aki.py", "prep_rx.py",
        "train_risk.py", "score_risk.py", "infer.py", "eval_model.py", "baseline.py", "check_env.py"]
PKGS = ["transformers", "peft", "accelerate", "bitsandbytes", "lightgbm", "scikit-learn", "pandas", "pyarrow",
        "numpy", "safetensors", "huggingface-hub", "tokenizers"]
INFO_FILES = ["eval_aki_risk.json", "eval_qlora_aki.json", "train_info_qlora_aki.json", "summary_aki.json",
              "summary_rx.json", "baseline.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def version(p):
    try:
        return metadata.version(p)
    except Exception:
        return None


def base_revision(model):
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model:
                return sorted(r.commit_hash for r in repo.revisions)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(C.HERE, "runs", "qlora_aki", "adapter"))
    ap.add_argument("--risk-dir", default=os.path.join(C.HERE, "runs", "aki_risk"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.adapter, "adapter_config.json")):
        raise SystemExit("[오류] 어댑터가 없습니다: %s  (3_처방모델_학습.bat 을 먼저 끝내세요)" % a.adapter)
    if not os.path.exists(os.path.join(a.risk_dir, "model.txt")):
        raise SystemExit("[오류] 위험 예측 모델이 없습니다: %s  (2_AKI위험도_학습.bat 을 먼저 실행하세요)" % a.risk_dir)
    with open(os.path.join(a.adapter, "adapter_config.json"), encoding="utf-8") as f:
        acfg = json.load(f)
    base = acfg.get("base_model_name_or_path") or "Qwen/Qwen3.5-9B"
    if os.path.isabs(str(base)) or os.sep in str(base).replace("/", ""):
        base = "Qwen/Qwen3.5-9B"

    out = a.out or os.path.join(C.HERE, "전달패키지_%s" % time.strftime("%Y%m%d"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "info"))
    for fn in CODE:
        shutil.copy2(os.path.join(C.HERE, fn), os.path.join(out, fn))
    for fn in ("README_받는분.md", "run_받는쪽.sh", "run_받는쪽.bat"):
        src = os.path.join(C.HERE, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, fn))
    ad_out = os.path.join(out, "runs", "qlora_aki", "adapter")
    shutil.copytree(a.adapter, ad_out, ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt", "*.bin.tmp"))
    rk_out = os.path.join(out, "runs", "aki_risk")
    os.makedirs(rk_out)
    for fn in ("model.txt", "config.json"):
        shutil.copy2(os.path.join(a.risk_dir, fn), os.path.join(rk_out, fn))
    for fn in INFO_FILES:
        p = os.path.join(C.RESULTS, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, "info", fn))

    vers = {p: version(p) for p in PKGS}
    try:
        import torch
        vers["torch"] = torch.__version__
        vers["torch_cuda"] = torch.version.cuda
    except Exception:
        vers["torch"] = version("torch")
    with open(os.path.join(out, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("# 보낸 쪽 환경과 같은 버전 (torch 는 서버의 CUDA 에 맞춰 따로 설치: 보낸 쪽 %s, CUDA %s)\n"
                % (vers.get("torch"), vers.get("torch_cuda")))
        for p in PKGS:
            if vers.get(p):
                f.write("%s==%s\n" % (p, vers[p]))
    files = {}
    for root, _, fs in os.walk(os.path.join(out, "runs")):
        for fn in fs:
            p = os.path.join(root, fn)
            files[os.path.relpath(p, out).replace(os.sep, "/")] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    info = {"created": time.strftime("%Y-%m-%d %H:%M"), "base_model": base, "base_model_revision": base_revision(base),
            "lora": {k: acfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules")},
            "python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "mimic_root_at_sender": C.MIMIC, "files": files,
            "note": "환자 단위 자료(data/, _work/)는 들어 있지 않음. 받는 쪽에서 같은 코드로 전처리해야 함"}
    with open(os.path.join(out, "info", "versions.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 환자 단위 자료가 섞이지 않았는지 확인
    bad = [os.path.relpath(os.path.join(r, fn), out) for r, _, fs in os.walk(out) for fn in fs
           if fn.endswith((".jsonl", ".parquet")) or "_work" in r or os.sep + "data" + os.sep in r + os.sep]
    if bad:
        shutil.rmtree(out)
        raise SystemExit("[오류] 환자 단위 파일이 섞여 패키지를 지웠습니다: %s" % bad[:5])
    size = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fs in os.walk(out) for fn in fs)
    print("[완료] %s (%.0f MB)" % (out, size / 1e6))
    if not a.no_zip:
        z = shutil.make_archive(out, "zip", os.path.dirname(out), os.path.basename(out))
        print("[완료] %s (%.0f MB)" % (z, os.path.getsize(z) / 1e6))


if __name__ == "__main__":
    main()
