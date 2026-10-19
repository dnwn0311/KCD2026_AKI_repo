# -*- coding: utf-8 -*-
import json, os, re, time
import numpy as np, pandas as pd
import train_risk_team as R

TD = '_work/team_fixed'
OUT = 'noleak_out'

L1_PAT = re.compile(r'^(creatinine|creatinine_wb|urine_ml|urine_creat|egfr_|bun_|tr_cr_|tr_bun_|tr_uo_|tr_urine_ml_24h)')
L2_PAT = re.compile(r'^(k_|na_|cl_|hco3_|mg_|phosphate_|uric_acid_|tr_k_delta|tr_hco3_delta|fluid_balance_ml|intake_ml|drain_ml|stool_ml|'
                    r'other_output_ml|tr_intake_24h|tr_fluid_bal_24h|weight_daily_|dialysis_hours|crrt_hours|ultrafiltrate|dial_|'
                    r'n_dialysis_sessions_before_t0|first_dialysis_hour|atc_C03C_loop_diuretic|atc3_C03)')


def load():
    cfg = json.load(open('runs/aki_risk_team_trend/config.json', encoding='utf-8'))
    feats0, cats = cfg['features'], cfg['categorical']
    wide = pd.read_parquet(TD + '/wide.parquet')
    tf = pd.read_parquet(TD + '/trend_features.parquet')
    nf = pd.read_parquet('_work/noleak/new_feats.parquet')
    spl = pd.read_parquet(TD + '/splits.parquet', columns=['stay_id', 'split_random'])
    cols = ['stay_id', 'subject_id', 'label_aki_next_24h'] + [c for c in feats0 if c in wide.columns]
    df = wide[cols].merge(tf, on='stay_id', how='left').merge(nf, on='stay_id', how='left').merge(spl, on='stay_id')
    df = df.rename(columns={'label_aki_next_24h': 'label', 'split_random': 'split'})
    df['split'] = df['split'].astype(str).replace({'validation': 'val'})
    for c in cats:
        vals = sorted(df.loc[df.split == 'train', c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=vals)
    nfc = [c for c in nf.columns if c != 'stay_id']
    for c in feats0 + nfc:
        if c not in cats:
            if df[c].dtype == bool:
                df[c] = df[c].astype('int8')
            elif df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors='coerce')
    df['label'] = df['label'].astype(int)
    # 학습 구간에서 거의 비어 있는 새 특징(95% 초과 결측)은 제외
    miss = df.loc[df.split == 'train', nfc].isna().mean()
    nfc = [c for c in nfc if miss[c] <= 0.95]
    l1 = [c for c in feats0 if L1_PAT.match(c)]
    l2 = [c for c in feats0 if L2_PAT.match(c) and c not in l1]
    sets = {
        'A_전체280': feats0,
        'B_정의직결제외': [c for c in feats0 if c not in l1],
        'C_엄격제외': [c for c in feats0 if c not in l1 and c not in l2],
    }
    new_nosqrt = [c for c in nfc if not c.endswith('_sqrt')]
    sets['D_B+새특징'] = sets['B_정의직결제외'] + nfc
    sets['E_C+새특징'] = sets['C_엄격제외'] + nfc
    sets['D2_B+새특징(제곱근제외)'] = sets['B_정의직결제외'] + new_nosqrt
    sets['E2_C+새특징(제곱근제외)'] = sets['C_엄격제외'] + new_nosqrt
    return df, sets, cats, l1, l2, nfc


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)
