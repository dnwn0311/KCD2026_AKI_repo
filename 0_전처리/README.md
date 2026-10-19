# 0_전처리 · AKI 위험 모델(추세 포함)의 입력을 만드는 코드

추세 특징을 더한 위험 모델의 입력은 **네 단계**로 만들어집니다. 이 폴더에는 1단계(팀원 전처리 파이프라인)가 들어 있고, 2~3단계 코드는 `../1_팀위험모델/`에 있습니다.

```
MIMIC-IV 3.1 원본 CSV
   │  ① 팀원 전처리 파이프라인  (이 폴더: team_pipeline/scripts/01~12)
   ▼
cohort_final.parquet · wide.parquet · hourly.parquet · splits.parquet · default_features.json
   │  ② 약물 플래그 수정      (../1_팀위험모델/patch_drug_flags.py)
   ▼
_work/team_fixed/            (NSAID·ACEi/ARB 플래그를 성분명 기준으로 바로잡은 복사본)
   │  ③ 추세 특징 26개 생성    (../1_팀위험모델/train_risk_trend.py 의 build_trend, 입력은 hourly.parquet)
   ▼
_work/team_fixed/trend_features.parquet
   │  ④ 학습·평가              (../1_팀위험모델/train_risk_trend.py)
   ▼
runs/aki_risk_team_trend/ · results/eval_team_risk_trend.json
```

## 폴더 구성

| 파일 | 역할 |
|---|---|
| `team_pipeline/` | 팀원 전처리 코드 그대로 (`scripts/`, `config.yaml`, `docs/`, `requirements.txt`). 데이터는 없음 |
| `wsl_team_pipeline.sh` | 위 파이프라인을 WSL에서 한 번에 실행하는 래퍼 (코드 복사, 가상환경, MIMIC 연결, 단계 실행, 숫자 확인) |
| `team_link_mimic.py` | 파일 이름·압축 여부가 제각각인 MIMIC 폴더를 팀원 코드가 읽는 위치에 연결 |
| `team_check.py` | 실행 결과가 팀원 보고서 숫자와 같은지 확인 (11,000건 / 환자 10,211명 / 양성 1,733건) |

## ① 팀원 전처리 파이프라인 (`team_pipeline/scripts/`)

번호 순서대로 실행합니다. 한 번에 돌리는 스크립트는 `run_all.sh`입니다.

| 단계 | 파일 | 하는 일 | 만드는 것 | 시간(대략) |
|---|---|---|---|---|
| 01 | `01_csv_to_parquet.py` | MIMIC-IV CSV를 Parquet으로 변환 | `data/mimiciv/3.1_parquet/` | 20분~1시간 |
| 02 | `02_build_base_cohort.py` | 입원당 첫 ICU 입실, 18세 이상, 48시간 이상 재원 | `base_cohort.parquet`, `flow_base.csv` | 수 초 |
| 03 | `03_add_esrd_flag.py` | 말기콩팥병·만성 투석 표시 | `cohort_stage2.parquet` | 1분 |
| 04 | `04_build_labels.py` | KDIGO 라벨 (크레아티닌 또는 소변량) | `cohort_stage3.parquet` | 1~2분 |
| 05 | `05_build_final_cohort.py` | 남은 제외 규칙 적용 | `cohort_final.parquet`, `flow_final.csv` | 수 초 |
| 06 | `06_build_drug_map.py` | 처방을 ATC 약물군으로 매핑 (**인터넷** 필요) | `data/external/drug_map.parquet`, `rxnav_cache.json` | 처음 1시간, 캐시가 있으면 수 초 |
| **07** | **`07_build_features.py`** | 시간별 표와 1입실 1행 표 | **`hourly.parquet`**, **`wide.parquet`** | 1분 미만 |
| 08 | `08_build_splits.py` | 환자 단위 70/15/15 분할 (시드 42) | `splits.parquet` | 수 초 |
| 09 | `09_build_feature_dictionary.py` | 피처 사전 | `docs/feature_dictionary*.md` | 수 초 |
| 10 | `10_build_default_features.py` | 기본 모델 입력 열 목록 | `default_features.json` | 수 초 |
| 11 | `11_export_csv.py` | 작은 표를 CSV로 | `*.csv` | 수 초 |
| 12 | `12_build_standalone_report.py` | 독립 보고서 | `PREPROCESSING_REPORT.md` | 수 초 |

위험 모델 입력에 필요한 것은 01~08과 10입니다. 09, 11, 12는 문서화·내보내기라서 건너뛰어도 됩니다. 그 밖의 `00_make_sample_csvs.py`(작은 샘플 CSV), `compare_runs.py`(두 실행 비교), `make_share_zip.py`(공유용 zip)는 선택 도구입니다.

### 코호트와 라벨 (설정은 `team_pipeline/config.yaml`)

- 과제: 입실 48시간(`time_zero_hours`) 시점에서 이후 24시간(`horizon_hours`) 안의 첫 AKI를 예측합니다.
- 입실 흐름 (`3_결과_집계/전처리/flow_*.csv`):

| 규칙 | 남은 입실 |
|---|---|
| 전체 ICU 입실 | 94,458 |
| 입원당 첫 입실 | 85,242 |
| 18세 이상 | 85,242 |
| ICU 재원 48시간 이상 | 40,746 |
| 체중 있음 | 40,619 |
| 입실 0~48시간 크레아티닌 2개 이상 | 39,697 |
| time zero 이전에 AKI 시작하지 않음 | 17,646 |
| 결과를 알 수 있음 (구간 안 AKI 발생, 또는 구간 끝까지 재원) | 11,136 |
| ESRD가 아닌데 time zero 이전에 ICU에서 투석 시작한 입실 제외 | 11,086 |
| 라벨 0이면서 구간 안 크레아티닌·소변 자료가 없는 입실 제외 | **11,000** (환자 10,211명, 양성 1,733건, 15.8%) |

- 라벨 1 = 구간 안에서 KDIGO 기준 중 하나가 참: ① 크레아티닌이 48시간 이내 0.3 mg/dL 이상 상승, ② 7일 기준값의 1.5배 이상, ③ 소변량이 6시간 연속 0.5 mL/kg/h 미만. 라벨용 크레아티닌은 혈청(itemid 50912)만 씁니다. 혈액가스 크레아티닌(52024)은 특징에만 들어갑니다.
- 결측은 채우지 않고 그대로 두며, `_obs`(측정 여부)와 `_hsl`(마지막 측정 후 경과 시간) 열을 함께 만듭니다. 허용 범위를 벗어난 값은 결측으로 바꾸고 개수를 `n_outlier_*`에 기록합니다.
- 소변량 속도는 기록 간격이 6시간 이하일 때만 계산하고, 각 기록을 그 간격이 걸친 시간들에 나눠 담습니다. 체중은 입실 시 체중(없으면 time zero 이전의 첫 체중)을 씁니다.
- 열 이름과 설명은 `team_pipeline/docs/feature_dictionary_kr.md`, MIMIC itemid는 `docs/itemids_kr.md`, 코호트 규칙은 `docs/cohort_definition_kr.md`에 있습니다.

## ②~③ 약물 플래그 수정과 추세 특징 (`../1_팀위험모델/`)

- `patch_drug_flags.py`: 06단계의 ATC 코드에는 복합제 분류가 섞여 있어 diphenhydramine만 처방받아도 NSAID 플래그가, amlodipine만 처방받아도 ACEi/ARB 플래그가 1이 됩니다. 이 두 플래그(와 `atc3_M01A`, `n_kidney_relevant_classes`)를 성분명 기준으로 다시 만든 복사본을 `_work/team_fixed/`에 씁니다. 원본은 건드리지 않습니다.
- `train_risk_trend.py`의 `build_trend()`: `hourly.parquet`의 0~47시간 자료로 추세 특징 26개를 만들어 `trend_features.parquet`에 저장합니다 (크레아티닌 9, BUN·칼륨·중탄산 4, 소변량 7, 평균동맥압 2, 수액·소변·수분 균형 3, 승압제 1).

## 실행 방법 (WSL)

```bash
# 이 폴더(0_전처리)에서. MIMIC 원본이 /mnt/d/MIMIC 에 있는 경우
bash wsl_team_pipeline.sh
#   MIMIC_ROOT=/mnt/d/MIMIC   MIMIC 폴더 지정
#   AKI_PROJ=~/aki_pipeline   작업 폴더 (기본값)
#   FROM=02                   해당 단계부터 다시 실행 (01은 오래 걸림)
#   STOP_AFTER=05             코호트까지만 먼저 실행해 숫자 확인
```

- 결과는 `~/aki_pipeline/data/` 아래에 생깁니다. 이어서 `../1_팀위험모델/run_team_model.sh`가 `PIPE=~/aki_pipeline`에서 필요한 파일을 읽어 ②~④를 실행합니다.
- 06단계는 `rxnav_cache.json`이 없으면 인터넷(NLM RxNav)으로 약물 매핑을 만들어 1시간쯤 걸립니다. 같은 매핑을 보장하려면 기존 실행의 `data/external/rxnav_cache.json`을 `team_pipeline/data/external/`에 넣어 두세요 (이 폴더에는 약 정보 캐시만 넣고 환자 정보는 없습니다).
- 05단계 뒤 `team_check.py`가 **입실 11,000 / 환자 10,211 / 양성 1,733**과 비교합니다. 다르면 MIMIC 파일이 덜 받아졌거나 버전이 다른 것이므로 다음 단계로 넘어가지 말고 확인하세요.
- 팀원 방식대로 직접 돌리려면 프로젝트 루트에서 `PY=.venv/bin/python bash scripts/run_all.sh` (CSV는 `data/mimiciv/3.1/`에 둠, 01단계를 건너뛰려면 `SKIP_01=1`).
- 래퍼는 01단계 복사본에만 DuckDB 메모리 상한을 추가합니다 (결과는 같음).
- 메모리·디스크를 많이 씁니다. WSL 여유 디스크 40GB 이상(변환 결과 10~20GB), 처방 LLM 학습이 도는 중에는 실행하지 마세요 (스크립트가 멈춥니다).

## 자료 취급

환자 단위 자료(`data/`, `_work/`)는 이 PC 밖으로 내보내지 않고 외부 온라인 LLM·API에도 보내지 않습니다. 공유는 집계 결과(`flow_*.csv`, `results/`)만 합니다.
