# -*- coding: utf-8 -*-
"""
KCD 중환자실 처방 변경 제안 모델 - 공통 설정과 함수

환자 단위 데이터는 이 PC 안에서만 처리합니다.
results/ 폴더에는 집계 결과만 저장하며, 10 미만의 건수는 "<10" 으로 표시합니다.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_work")        # 환자 단위 중간 파일 (PC 밖으로 내보내지 않음)
DATA = os.path.join(HERE, os.environ.get("KCD_DATA_DIR", "data"))   # 학습용 JSONL (환자 단위, PC 밖으로 내보내지 않음). 팀원 기준 데이터는 KCD_DATA_DIR=data_team
RESULTS = os.path.join(HERE, "results")   # 집계 결과만

MIMIC = os.environ.get("MIMIC_ROOT", r"D:\MIMIC")
HOSP = os.path.join(MIMIC, "mimiciv", "hosp")
ICU = os.path.join(MIMIC, "mimiciv", "icu")

# 후보 경로 중 처음 존재하는 파일을 사용 (PhysioNet 에서 직접 받은 파일 우선)
FILE_CANDIDATES = {
    "patients": [os.path.join(HOSP, "patients.csv.gz"), os.path.join(MIMIC, "patients.csv.gz")],
    "admissions": [os.path.join(HOSP, "admissions.csv.gz"), os.path.join(MIMIC, "admissions.csv.gz")],
    "labevents": [os.path.join(HOSP, "labevents.csv.gz")],
    "d_labitems": [os.path.join(HOSP, "d_labitems.csv.gz")],
    "prescriptions": [os.path.join(HOSP, "prescriptions.csv.gz")],
    # chartevents: 직접 받은 파일이 완성되면 그것을, 아니면 서버 크기와 바이트 수가 일치한 사본을 사용
    "chartevents": [os.path.join(ICU, "chartevents.csv.gz"), os.path.join(MIMIC, "chartevents.csv-001-013.gz")],
    "d_items": [os.path.join(ICU, "d_items.csv.gz"), os.path.join(MIMIC, "d_items.csv.gz")],
    # 처방 중단·변경 이력 (D/C, Change 오더). 없으면 중단 정답을 만들 수 없음
    "poe": [os.path.join(HOSP, "poe.csv.gz"), os.path.join(MIMIC, "poe.csv.gz")],
    # 중환자실 체류 기간. 없으면 chartevents 의 stay_id 별 첫·마지막 기록 시각으로 대신함
    "icustays": [os.path.join(ICU, "icustays.csv.gz"), os.path.join(MIMIC, "icustays.csv.gz")],
}
CHARTEVENTS_EXPECTED_BYTES = 3502392765  # PhysioNet 서버의 chartevents.csv.gz 크기


def _search(root, name, max_depth=5):
    """MIMIC 폴더 구조가 달라도 찾도록 root 아래에서 파일 이름으로 검색 (예: mimiciv/3.1/hosp/...)"""
    if not os.path.isdir(root):
        return None
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        if name in files:
            return os.path.join(d, name)
    return None


def find_file(key, required=True):
    for p in FILE_CANDIDATES[key]:
        if os.path.exists(p):
            return p
    for p in FILE_CANDIDATES[key]:  # 표준 위치에 없으면 MIMIC_ROOT 아래에서 같은 이름을 찾음
        hit = _search(MIMIC, os.path.basename(p))
        if hit:
            return hit
    if required:
        raise SystemExit("[오류] %s 파일을 찾지 못했습니다. MIMIC_ROOT=%s 아래를 확인하세요. 확인한 위치: %s"
                         % (key, MIMIC, FILE_CANDIDATES[key]))
    return None


# ---------------------------------------------------------------------------
# 검사 항목 (MIMIC-IV itemid). d_labitems 의 이름과 대조해 맞지 않으면 제외합니다.
# (key, 한글명, 단위, 정상 하한, 정상 상한, itemids, 이름에 들어 있어야 할 단어)
LABS = [
    ("potassium", "칼륨", "mEq/L", 3.5, 5.0, [50971, 50822], "potassium"),
    ("sodium", "나트륨", "mEq/L", 135, 145, [50983, 50824], "sodium"),
    ("chloride", "염소", "mEq/L", 98, 107, [50902, 50806], "chloride"),
    ("bicarbonate", "중탄산", "mEq/L", 22, 29, [50882], "bicarbonate"),
    ("bun", "요소질소", "mg/dL", 7, 20, [51006], "urea"),
    ("creatinine", "크레아티닌", "mg/dL", 0.5, 1.2, [50912], "creatinine"),
    ("glucose", "혈당", "mg/dL", 70, 180, [50931, 50809], "glucose"),
    ("magnesium", "마그네슘", "mg/dL", 1.7, 2.4, [50960], "magnesium"),
    ("phosphate", "인", "mg/dL", 2.5, 4.5, [50970], "phosph"),
    ("calcium", "칼슘", "mg/dL", 8.4, 10.2, [50893], "calcium"),
    ("hemoglobin", "혈색소", "g/dL", 12, 17, [51222, 50811], "hemoglobin"),
    ("wbc", "백혈구", "K/uL", 4, 11, [51301, 51300], "white"),
    ("platelet", "혈소판", "K/uL", 150, 400, [51265], "platelet"),
    ("lactate", "젖산", "mmol/L", 0.5, 2.0, [50813], "lactate"),
    ("inr", "INR", "", 0.9, 1.2, [51237], "inr"),
]
LAB_KEYS = [x[0] for x in LABS]
LAB_INFO = {x[0]: x for x in LABS}

# 활력징후 (chartevents itemid, 중환자실 기록). 범위를 벗어난 값은 측정 오류로 보고 제외합니다.
# (key, 한글명, 단위, itemids, 이름 단어, 허용 하한, 허용 상한)
VITALS = [
    ("heart_rate", "심박수", "회/분", [220045], "heart rate", 20, 250),
    ("sbp", "수축기혈압", "mmHg", [220179, 220050], "systolic", 40, 300),
    ("dbp", "이완기혈압", "mmHg", [220180, 220051], "diastolic", 10, 200),
    ("map", "평균동맥압", "mmHg", [220181, 220052], "mean", 20, 200),
    ("resp_rate", "호흡수", "회/분", [220210, 224690], "respiratory rate", 3, 70),
    ("spo2", "산소포화도", "%", [220277], "o2 saturation", 50, 100),
    ("temp_c", "체온", "℃", [223762, 223761], "temperature", 30, 45),  # 223761(화씨)는 섭씨로 변환
]
VITAL_KEYS = [x[0] for x in VITALS]
VITAL_INFO = {x[0]: x for x in VITALS}
TEMP_F_ITEMID = 223761

# ---------------------------------------------------------------------------
# 감지할 이상 상황. 활력징후는 1시간 안에 두 번 연속 이상일 때만 감지 (측정 오류 방지)
# (이름, 한글명, 종류, 항목, 비교, 기준값)
SCENARIOS = [
    ("hypokalemia", "저칼륨혈증", "lab", "potassium", "<", 3.5),
    ("hyperkalemia", "고칼륨혈증", "lab", "potassium", ">", 5.5),
    ("hypomagnesemia", "저마그네슘혈증", "lab", "magnesium", "<", 1.6),
    ("hypophosphatemia", "저인산혈증", "lab", "phosphate", "<", 2.0),
    ("hyperglycemia", "고혈당", "lab", "glucose", ">", 250),
    ("hypoglycemia", "저혈당", "lab", "glucose", "<", 70),
    ("hypotension", "저혈압", "vital", "map", "<", 65),
    ("tachycardia", "빈맥", "vital", "heart_rate", ">", 130),
    ("hypoxemia", "저산소증", "vital", "spo2", "<", 88),
    ("fever", "발열", "vital", "temp_c", ">", 38.3),
]
SCENARIO_KR = {s[0]: s[1] for s in SCENARIOS}
SCENARIO_KR["aki"] = "급성 신손상"
SCENARIO_KR["aki_risk"] = "AKI 위험 예측"

# ---------------------------------------------------------------------------
# AKI (KDIGO 크레아티닌 기준). 소변량 기준은 outputevents 가 없어 쓰지 않음
AKI_ABS_RISE = 0.3          # 48시간 안 최저값보다 0.3 mg/dL 이상 상승
AKI_ABS_HOURS = 48
AKI_REL_RISE = 1.5          # 7일 안 최저값(기저치)의 1.5배 이상
AKI_REL_DAYS = 7
AKI_ICU_HOURS = 48          # 처방 모델: 입실 48시간 이후에 감지된 AKI 만 사용 / 위험 예측: 첫 예측 시점
# 위험 예측 (이동 창): 입실 48시간부터 24시간마다 예측. 직전 48시간 데이터로 이후 24시간 안의 새 AKI 를 예측
RISK_FIRST_HOURS = 48       # 첫 예측 시점 (입실 후 시간)
RISK_STEP_HOURS = 24        # 예측 간격
RISK_WINDOW_HOURS = 48      # 입력 창: 예측 시점 직전 48시간의 검사·활력징후·처방 (크레아티닌 기저치는 7일)
RISK_HORIZON = 24           # 예측 구간: 예측 시점 이후 24시간
RISK_MAX_POINTS = 14        # 체류당 예측 시점 상한 (입실 후 약 16일까지). 학습에는 체류당 가중치(1/시점 수)도 씀
AKI_MAX_EVENTS = 2          # 처방 모델: 48시간 이후 중환자실에서 감지된 AKI, 체류당 최대 횟수 (24시간 간격)


def kdigo_stage(value, baseline):
    """크레아티닌 기준 KDIGO 단계 (1~3). 기저치가 없으면 None"""
    if baseline is None or not baseline or baseline != baseline:
        return None
    r = value / baseline
    if r >= 3.0 or value >= 4.0:
        return 3
    if r >= 2.0:
        return 2
    return 1


# AKI 에 영향을 주는 약물 (정규화한 약 이름에 아래 단어가 들어 있으면 해당). 의료진 검토용 초안
AKI_DRUG_GROUPS = [
    ("신독성", ["ketorolac", "ibuprofen", "naproxen", "celecoxib", "diclofenac", "indomethacin", "meloxicam",
             "gentamicin", "tobramycin", "amikacin", "vancomycin", "amphotericin", "colistin", "polymyxin",
             "acyclovir", "foscarnet", "cidofovir", "tacrolimus", "cyclosporine", "methotrexate", "cisplatin",
             "lithium"]),
    ("신혈류", ["lisinopril", "captopril", "enalapril", "ramipril", "benazepril", "quinapril", "losartan",
             "valsartan", "irbesartan", "candesartan", "olmesartan", "sacubitril"]),
    ("이뇨·수액", ["furosemide", "bumetanide", "torsemide", "hydrochlorothiazide", "chlorthalidone", "metolazone",
               "spironolactone", "acetazolamide", "sodium chloride 0.9%", "lactated ringers", "albumin"]),
    ("신배설 용량조절", ["metformin", "enoxaparin", "gabapentin", "pregabalin", "cefepime", "meropenem",
                  "piperacillin-tazobactam", "levetiracetam", "famotidine", "digoxin", "apixaban", "rivaroxaban",
                  "dabigatran", "fluconazole", "ciprofloxacin", "levofloxacin", "sulfamethoxazole", "trimethoprim",
                  "morphine", "glyburide"]),
]
VASOPRESSORS = ["norepinephrine", "vasopressin", "phenylephrine", "epinephrine", "dopamine"]
_AKI_CAT_CACHE = {}


def aki_category(name):
    """약 이름 -> AKI 관련 분류 (해당 없으면 None)"""
    if not isinstance(name, str):
        return None
    if name not in _AKI_CAT_CACHE:
        cat = None
        for c, words in AKI_DRUG_GROUPS:
            if any(w in name for w in words):
                cat = c
                break
        _AKI_CAT_CACHE[name] = cat
    return _AKI_CAT_CACHE[name]

# 평가용 "상황별 대표 반응 약물" (임상 정답이 아니라 성능 비교를 위한 휴리스틱)
TARGET_DRUGS = {
    "hypokalemia": ["potassium chloride", "potassium phosphate"],
    "hyperkalemia": ["calcium gluconate", "insulin (short-acting)", "sodium polystyrene sulfonate",
                     "patiromer", "dextrose 50%", "sodium bicarbonate", "furosemide", "albuterol"],
    "hypomagnesemia": ["magnesium sulfate", "magnesium oxide"],
    "hypophosphatemia": ["potassium phosphate", "sodium phosphate", "phosphorus (oral)"],
    "hyperglycemia": ["insulin (short-acting)", "insulin (basal)"],
    "hypoglycemia": ["dextrose 50%", "glucagon", "dextrose"],
    "hypotension": ["norepinephrine", "phenylephrine", "vasopressin", "epinephrine", "dopamine",
                    "albumin", "lactated ringers", "sodium chloride 0.9%"],
    "tachycardia": ["metoprolol", "diltiazem", "amiodarone", "esmolol", "digoxin"],
    "hypoxemia": ["furosemide", "albuterol", "ipratropium-albuterol", "methylprednisolone"],
    "fever": ["acetaminophen", "vancomycin", "cefepime", "piperacillin-tazobactam", "meropenem",
              "ceftriaxone"],
}

WINDOW_HOURS = 12          # 이상 감지 후 이 시간 안의 처방 변경(신규·감량·증량·중단)을 정답으로 사용
LAB_LOOKBACK_HOURS = 24    # 상태 요약: 검사값은 24시간 이내 최신값
VITAL_LOOKBACK_HOURS = 6   # 상태 요약: 활력징후는 6시간 이내 최신값
DEDUP_HOURS = 24           # 같은 입원·같은 상황은 24시간 간격으로만 다시 감지
MAX_PER_ADMISSION = 3      # 같은 입원·같은 상황 최대 감지 수
MAX_ACTIVE_MEDS = 20
MAX_LABEL_MEDS = 10
ICU_ONLY = True            # 중환자실 체류 중에 감지된 경우만 사용
DOSE_CHANGE_RATIO = 0.10   # 하루 용량이 10% 이상 줄면 감량, 늘면 증량

# 정답·출력의 처방 변경 종류 (JSON 키 순서도 이대로)
ACTIONS = ["신규처방", "감량", "증량", "중단"]

SYSTEM_PROMPT = (
    "당신은 중환자실 환자의 급성 신손상(AKI) 위험 예측 결과나 AKI 감지 정보를 보고, AKI 에 영향을 주는 약물"
    "(신독성 약, 신혈류에 영향을 주는 약, 이뇨제·수액, 신장으로 배설되어 용량 조절이 필요한 약)의 "
    "처방 변경 초안을 의사에게 제안하는 보조 AI입니다. "
    "제안은 의사가 최종 확인하기 전의 초안이며, 환자 상태 요약과 위험 예측 결과만 근거로 합니다. "
    "반드시 JSON 한 개로만 답하세요: {\"상황\": 문자열, \"신규처방\": [새로 시작할 약], "
    "\"감량\": [용량을 줄일 투여 중인 약], \"증량\": [용량을 늘릴 투여 중인 약], \"중단\": [끊을 투여 중인 약]}. "
    "감량·증량·중단은 현재 투여 중인 AKI 관련 약 중에서만 고르고, 바꿀 약이 없으면 모두 빈 목록으로 둡니다."
)
RISK_ASK = "이 환자의 AKI 위험을 고려해, AKI 와 관련된 약물의 처방 변경(신규 시작, 감량, 증량, 중단) 초안을 제안하세요."
AKI_ASK = "이 환자에게 AKI 와 관련된 약물의 처방 변경(신규 시작, 감량, 증량, 중단) 초안을 제안하세요."
TIER_KR = {"red": "빨강(고위험)", "yellow": "노랑(주의)", "green": "초록(낮음)"}


_STAT_KR = {"last": "최근", "min": "최저", "max": "최고", "mean": "평균"}
_FEAT_KR = {
    "age": "나이", "male": "남성", "icu_hours_at_T": "중환자실 경과 시간", "hosp_hours_at_T": "입원 경과 시간",
    "admission_type": "입원 유형", "admission_location": "입원 경로",
    "cr_last": "크레아티닌 최근값", "cr_base7d": "크레아티닌 기저치(7일 최저)", "cr_last_over_base": "크레아티닌/기저치",
    "cr_last_minus_base": "기저치 대비 크레아티닌 상승", "cr_last_minus_min48h": "48시간 최저값 대비 크레아티닌 상승",
    "cr_change48h": "48시간 크레아티닌 변화", "cr_first48h": "48시간 전 크레아티닌", "cr_min48h": "48시간 크레아티닌 최저",
    "cr_max48h": "48시간 크레아티닌 최고", "cr_n48h": "48시간 크레아티닌 검사 횟수",
    "cr_hours_since_last": "마지막 크레아티닌 검사 후 시간",
    "vit_map_lt65_n": "48시간 평균동맥압 65 미만 횟수", "vit_hr_gt100_n": "48시간 심박수 100 초과 횟수",
    "med_active_n": "투여 중인 약 수", "med_started_n": "48시간 내 시작한 약 수",
    "med_active_vancomycin": "반코마이신 투여 중", "med_active_piptazo": "피페라실린-타조박탐 투여 중",
    "med_active_nsaid": "NSAID 투여 중", "med_active_aminoglycoside": "아미노글리코사이드 투여 중",
    "med_active_loop": "루프 이뇨제 투여 중", "med_active_vasopressor": "승압제 투여 중",
    "med_vanc_piptazo": "반코마이신+피페라실린-타조박탐 병용",
}


def feature_text(name, value):
    """위험 요인 설명용: 특징 이름과 값을 한국어로 ('크레아티닌/기저치 1.4배', '승압제 투여 중')"""
    if name in _FEAT_KR:
        label = _FEAT_KR[name]
    elif name.startswith("med_active_") or name.startswith("med_started_"):
        cat = name.split("_", 2)[2]
        label = ("투여 중인 %s 약 수" if name.startswith("med_active_") else "48시간 내 시작한 %s 약 수") % cat
    elif name.startswith("lab_") or name.startswith("vit_"):
        key, stat = name[4:].rsplit("_", 1)
        info = LAB_INFO.get(key) if name.startswith("lab_") else VITAL_INFO.get(key)
        kr = info[1] if info else key
        label = "48시간 %s %s" % (kr, _STAT_KR.get(stat, stat))
    else:
        label = name
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "%s %s" % (label, value)
    if v != v:
        return "%s 기록 없음" % label
    if label.endswith("투여 중") or label.endswith("병용") or name == "male":
        return label if v >= 0.5 else "%s 아님" % label
    if name == "cr_last_over_base":
        return "%s %.1f배" % (label, v)
    if name.endswith("_n") or name.endswith("_n48h") or label.endswith("수") or label.endswith("횟수"):
        return "%s %d" % (label, round(v))
    if "hours" in name:
        return "%s %.0f시간" % (label, v)
    return "%s %s" % (label, ("%.1f" % v) if abs(v) < 100 else ("%.0f" % v))


def empty_label():
    return {k: [] for k in ACTIONS}

# ---------------------------------------------------------------------------
# 약물명 정규화
_PAREN = re.compile(r"\([^)]*\)")
_DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|gm|meq|ml|units?|unit|%|mmol|l|mg/ml|meq/l)\b", re.I)
_NUM = re.compile(r"\b\d+(\.\d+)?\b")
_FORM = re.compile(r"\b(tab|tabs|tablet|tablets|cap|caps|capsule|capsules|inj|injection|soln|solution|"
                   r"susp|suspension|oral|po|iv|ivpb|sr|er|xl|cr|dr|ec|liquid|syringe|vial|bag|premix|"
                   r"in\s+d5w|in\s+ns|in\s+sw|replacement\s+scale|sliding\s+scale|desensitization|"
                   r"\(?bulk\)?|ud|unit\s+dose|neb|nebulizer|hcl|sulfate\s+inj)\b", re.I)
EXCLUDE_PATTERNS = ["flush", "sterile water", "dressing", "lidocaine jelly", "heparin lock"]


def normalize_drug(name):
    """처방명 -> 학습용 약물명. 제외 대상이면 None."""
    if not isinstance(name, str):
        return None
    s = name.strip().lower().replace("*nf*", " ").strip()
    if not s:
        return None
    for pat in EXCLUDE_PATTERNS:
        if pat in s:
            return None
    # 의미가 중요한 것은 먼저 처리
    if "insulin" in s or "humalog" in s or "novolog" in s or "lantus" in s:
        if any(k in s for k in ("glargine", "lantus", "detemir", "nph", "degludec", "basaglar", "levemir")):
            return "insulin (basal)"
        return "insulin (short-acting)"
    if "dextrose 50" in s or "d50" in s:
        return "dextrose 50%"
    if "potassium chloride" in s or re.search(r"\bkcl\b", s) or "potassium chl" in s or "k-dur" in s:
        return "potassium chloride"
    if "potassium phos" in s or "k phos" in s or "k-phos" in s:
        return "potassium phosphate"
    if "sodium phos" in s:
        return "sodium phosphate"
    if "neutra-phos" in s or "phospha" in s and "neutral" in s:
        return "phosphorus (oral)"
    if "magnesium sulf" in s:
        return "magnesium sulfate"
    if "magnesium oxide" in s or "mag-ox" in s:
        return "magnesium oxide"
    if "polystyrene" in s or "kayexalate" in s:
        return "sodium polystyrene sulfonate"
    if "piperacillin" in s or "zosyn" in s:
        return "piperacillin-tazobactam"
    if "ipratropium" in s and "albuterol" in s:
        return "ipratropium-albuterol"
    if "lactated ringer" in s:
        return "lactated ringers"
    if ("sodium chloride 0.9" in s or s.startswith("0.9% sodium chloride") or s in ("ns", "normal saline")):
        return "sodium chloride 0.9%"
    s = _PAREN.sub(" ", s)
    s = _DOSE.sub(" ", s)
    s = _FORM.sub(" ", s)
    s = _NUM.sub(" ", s)
    s = re.sub(r"[^a-z\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    if len(s) < 3:
        return None
    if s.startswith("acetaminophen"):
        return "acetaminophen"
    if s.startswith("metoprolol"):
        return "metoprolol"
    if s.startswith("vancomycin"):
        return "vancomycin"
    if s.startswith("dextrose"):
        return "dextrose"
    return s


# ---------------------------------------------------------------------------
def subject_split(subject_id):
    """환자 단위 고정 분할: train 80 / val 10 / test 10"""
    h = (int(subject_id) * 2654435761) % 4294967296 % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def suppress(n):
    """집계 결과에 10 미만의 건수를 그대로 쓰지 않는다."""
    n = int(n)
    return n if n >= 10 else "<10"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm_list(v):
    if not isinstance(v, list):
        return []
    out = []
    for m in v:
        m = str(m).strip()
        if m:
            d = normalize_drug(m) or m.lower()
            if d not in out:
                out.append(d)
    return out


def parse_answer(text):
    """모델 출력에서 JSON 을 찾아 {신규처방, 감량, 증량, 중단} 목록을 돌려준다. 실패하면 None."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict) and any(k in obj for k in ACTIONS):
                            return {k: _norm_list(obj.get(k)) for k in ACTIONS}
                    except Exception:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def set_metrics(rows):
    """rows: [(scenario, true_dict, pred_dict_or_None)] -> 집계 지표

    - 전체: (변경 종류, 약) 쌍 기준 micro 정밀도·재현율·F1, 완전 일치율
    - 종류별: 신규처방·감량·증량·중단 각각의 micro 지표와 정답이 있는 사례 수
    - 대표약 재현율: 신규처방 중 상황별 대표 반응 약물을 맞힌 비율 (이전 버전과 비교용)
    """
    def agg(sub):
        n = len(sub)
        parsed = exact = 0
        tot = [0, 0, 0]
        per = {k: [0, 0, 0, 0] for k in ACTIONS}  # tp, fp, fn, 정답 있는 사례
        tgt_true = tgt_hit = tgt_pred = tgt_pred_ok = 0
        for scen, true, pred in sub:
            if pred is not None:
                parsed += 1
            pred = pred or empty_label()
            all_ok = True
            for k in ACTIONS:
                T, P = set(true.get(k, [])), set(pred.get(k, []))
                tp, fp, fn = len(T & P), len(P - T), len(T - P)
                per[k][0] += tp
                per[k][1] += fp
                per[k][2] += fn
                per[k][3] += int(bool(T))
                tot[0] += tp
                tot[1] += fp
                tot[2] += fn
                all_ok &= (T == P)
            exact += int(all_ok)
            tg = set(TARGET_DRUGS.get(scen, []))
            T, P = set(true.get("신규처방", [])), set(pred.get("신규처방", []))
            if T & tg:
                tgt_true += 1
                tgt_hit += int(bool(P & T & tg))
            if P & tg:
                tgt_pred += 1
                tgt_pred_ok += int(bool(P & tg & T))
        p, r, f = _prf(*tot)
        out = {"n": suppress(n), "parse_rate": round(parsed / n, 4) if n else None,
               "micro_precision": p, "micro_recall": r, "micro_f1": f,
               "exact_match": round(exact / n, 4) if n else None,
               "target_recall": round(tgt_hit / tgt_true, 4) if tgt_true else None,
               "target_precision": round(tgt_pred_ok / tgt_pred, 4) if tgt_pred else None,
               "target_cases": suppress(tgt_true), "by_action": {}}
        for k in ACTIONS:
            if sum(per[k][:3]) == 0:  # 정답도 예측도 없음
                out["by_action"][k] = {"precision": None, "recall": None, "f1": None, "cases_with_label": 0}
                continue
            p, r, f = _prf(*per[k][:3])
            out["by_action"][k] = {"precision": p, "recall": r, "f1": f, "cases_with_label": suppress(per[k][3])}
        return out

    out = {"overall": agg(rows), "by_scenario": {}}
    for s in SCENARIO_KR:
        sub = [r for r in rows if r[0] == s]
        if sub:
            out["by_scenario"][s] = agg(sub)
    return out
