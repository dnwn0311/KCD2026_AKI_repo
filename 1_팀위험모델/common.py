# -*- coding: utf-8 -*-
"""
위험 모델 스크립트 공통 설정 (원본 common.py 에서 위험 모델이 쓰는 부분만 옮긴 것)

환자 단위 파일은 _work/ 안에서만 만들어지고 읽힙니다 (이 폴더 밖으로 내보내지 않습니다).
results/ 에는 집계 결과만 저장하며, 10 미만의 건수는 "<10" 으로 가립니다.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_work")        # 환자 단위 중간 파일
RESULTS = os.path.join(HERE, "results")   # 집계 결과만


def suppress(n):
    """집계 결과에 10 미만의 건수를 그대로 쓰지 않는다."""
    n = int(n)
    return n if n >= 10 else "<10"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
