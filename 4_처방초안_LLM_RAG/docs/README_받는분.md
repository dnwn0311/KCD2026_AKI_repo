# KCD AKI 모델 전달 패키지 사용 안내

이 패키지에는 두 모델과 같은 입력을 만들기 위한 코드가 들어 있습니다. **환자 단위 자료는 들어 있지 않으므로**, 받는 분의 서버에서 본인 MIMIC-IV 로 전처리해야 합니다. 환자 분할 규칙이 고정되어 있어 같은 MIMIC 버전이면 보낸 쪽과 같은 test 환자가 만들어집니다.

| 들어 있는 것 | 위치 | 설명 |
|---|---|---|
| 처방 LLM 어댑터 | `runs/qlora_aki/adapter/` | Qwen3.5-9B 전용 LoRA. 다른 기본 모델(예: 검증용 27B)에는 붙지 않음 |
| 위험 예측 모델 | `runs/aki_risk/` | LightGBM. `config.json` 에 특징 순서, 확률 보정값, 신호등 기준값 |
| 코드 | `*.py` | 전처리·점수·초안 생성. 보낸 쪽이 학습에 쓴 코드와 같은 버전 |
| 재현 확인 정보 | `info/` | 라이브러리 버전(`versions.json`), 파일 체크섬, 보낸 쪽 성능 집계 |

## 1. 환경

- GPU 메모리 10GB 이상 (Qwen3.5-9B 4비트 추론). 검증용 LLM 을 같은 서버에 올리면 그만큼 더 필요
- Python 3.10 이상. torch 는 서버 CUDA 에 맞춰 먼저 설치한 뒤 `pip install -r requirements.txt`
- 기본 모델은 Hugging Face 에서 받음: `Qwen/Qwen3.5-9B` (보낸 쪽 리비전은 `info/versions.json`)
- 설치 확인: `python check_env.py`

## 2. 실행 (Linux: `bash run_받는쪽.sh`, Windows: `run_받는쪽.bat`)

```bash
export MIMIC_ROOT=/경로/MIMIC   # hosp, icu 폴더가 이 아래 어디에 있든 이름으로 찾습니다
python prep_extract.py          # 원본에서 필요한 부분 추출 (30분~1시간)
python prep_aki.py              # AKI 판정, 위험 예측 입력 생성
python score_risk.py            # 받은 위험 예측 모델로 점수 (다시 학습하지 않음)
python prep_rx.py               # 위험 예측 결과를 넣은 처방 LLM 입력 생성
python infer.py --split test    # 받은 어댑터로 처방 초안 생성
```

`score_risk.py` 가 끝나면 test 성능을 보낸 쪽 값과 비교해 보여 줍니다. 차이가 크면 MIMIC 버전이나 코드 버전이 다른 것입니다 (`results/score_risk_check.json`).

## 3. 검증 AI 에 넘길 파일

`_work/drafts_test.jsonl` 에 한 줄에 한 사례씩 저장됩니다.

| 필드 | 내용 |
|---|---|
| `case_id` | 사례 번호 (상황-입원번호-시각) |
| `scenario` | `aki_risk` = 위험 예측 시점(예방적 조정), `aki` = AKI 감지 시점(대응) |
| `risk` | 위험 예측 시점일 때 확률·신호등(`red`/`yellow`/`green`)·주요 요인 |
| `prompt` | 환자 상태 요약 (LLM 이 본 입력 그대로) |
| `draft` | 초안 `{"신규처방": [], "감량": [], "증량": [], "중단": []}` (해석 실패 시 `null`) |
| `actual` | 실제 기록된 처방 변경 (평가용 정답) |

약 이름은 정규화된 이름입니다 (예: `enoxaparin sodium`, `piperacillin-tazobactam`, `insulin (short-acting)`). 약물 DB 와 이름을 맞출 때 `common.py` 의 `normalize_drug`, AKI 관련 약물 목록은 `AKI_DRUG_GROUPS` 를 참고하세요.

## 4. 주의

- `data/`, `_work/` 의 파일과 `drafts_*.jsonl` 은 환자 단위 자료입니다. 연구실 서버 밖이나 외부 서비스(클라우드 LLM 포함)로 보내지 마세요. 검증 LLM 도 서버 안에서 돌려야 합니다
- 어댑터는 학습 데이터 일부를 기억할 수 있으므로 공개 저장소에 올리지 말고 MIMIC 접근 권한이 있는 팀원끼리만 공유하세요
- `train` 분할의 위험 점수는 이 모델이 학습한 데이터라 실제보다 좋게 나옵니다. 검증에는 `val`·`test` 만 쓰세요
- 결과는 의사가 확인하기 전의 초안이며, 과거 처방 관행을 따라 한 것이지 안전성을 보장하지 않습니다
