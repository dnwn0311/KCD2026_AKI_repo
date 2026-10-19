# -*- coding: utf-8 -*-
"""
학습 환경 점검. 종료 코드: 0 정상, 1 GPU/라이브러리 문제, 2 transformers 가 Qwen3.5 를 지원하지 않음
  python check_env.py           (전체 점검: 4비트 연산 시험 포함)
  python check_env.py --quick   (설치 직후 빠른 점검)
"""
import argparse
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    code = 0
    print("Python", sys.version.split()[0])
    try:
        import torch
        print("torch", torch.__version__, "| CUDA 빌드", torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print("GPU %s | 메모리 %.1f GB | 연산능력 %d.%d | bf16 %s"
                  % (p.name, p.total_memory / 1024 ** 3, p.major, p.minor, torch.cuda.is_bf16_supported()))
            x = torch.randn(256, 256, device="cuda")
            _ = (x @ x).sum().item()
            print("  GPU 연산 OK")
        else:
            print("[문제] GPU(CUDA)를 쓸 수 없습니다. RTX 50 시리즈는 CUDA 12.8 용 torch 가 필요합니다:")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu128")
            code = 1
    except Exception as e:
        print("[문제] torch:", e)
        return 1
    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        ok = "qwen3_5" in CONFIG_MAPPING_NAMES
        print("transformers", transformers.__version__, "| Qwen3.5 지원:", "예" if ok else "아니오")
        if not ok:
            code = max(code, 2)
    except Exception as e:
        print("[문제] transformers:", e)
        code = 1
    for mod in ("peft", "accelerate", "bitsandbytes", "pandas", "pyarrow"):
        try:
            m = __import__(mod)
            print(mod, getattr(m, "__version__", "?"))
        except Exception as e:
            print("[문제] %s: %s" % (mod, e))
            code = 1 if mod != "pyarrow" else code
    if not a.quick and code != 1:
        try:
            import torch
            import bitsandbytes as bnb
            lin = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
            y = lin(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
            print("  4비트(bitsandbytes) 연산 OK", tuple(y.shape))
        except Exception as e:
            print("[문제] 4비트 연산 실패:", e)
            code = 1
    # Qwen3.5 의 선형 어텐션(Gated DeltaNet) 가속 라이브러리 (없어도 동작하지만 느림)
    for mod, pkg in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        try:
            __import__(mod)
            print("%s 설치됨 (Qwen3.5 학습 가속)" % pkg)
        except Exception:
            print("%s 없음 - Qwen3.5 는 느린 기본 구현으로 동작합니다 (선택 사항)" % pkg)
    total, used, free = shutil.disk_usage(".")
    print("디스크 여유 (현재 드라이브): %.0f GB  (Qwen3.5-9B 내려받기에 약 20GB 필요)" % (free / 1024 ** 3))
    print("점검 결과:", {0: "정상", 1: "문제 있음", 2: "transformers 업데이트 필요"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
