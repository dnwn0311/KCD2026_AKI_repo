# -*- coding: utf-8 -*-
import time as _t; T0=_t.time()
"""순간값(snapshot) 신장 피처 제외 + 흐름(플로우) 피처 추가 — 교차검증(환자 단위 5-겹 x 3회) 비교"""
import json, re, sys, time, warnings
import numpy as np, pandas as pd
src = open('noleak_groups.py').read().split("B = sets['B_정의직결제외']")[0].replace("for k, v in G.items(): print(k, len(v))", "")
exec(src)
fl = pd.read_parquet('_work/noleak/flow_feats.parquet'); df = df.merge(fl, on='stay_id', how='left')
FL = [c for c in fl.columns if c != 'stay_id']
A = sets['A_전체280']
SNAP_RE = re.compile(r'^(creatinine|urine_creat|urine_ml|egfr_|bun_|tr_uo_rate|tr_uo_obs|tr_uo_low|tr_urine_ml|tr_cr_n$|tr_cr_hsl|tr_bun_cr_ratio)')
snap = [c for c in A if SNAP_RE.match(c)]
CHG = [c for c in A if c.startswith(('tr_cr_', 'tr_bun_delta', 'tr_uo_trend')) and c not in snap]
S0 = [c for c in A if c not in snap]
vs = G['vit_slope']
uo_abs = [c for c in FL if re.match(r'nf4_uo_bin', c)]
uo = [c for c in FL if c.startswith('nf4_uo')]; cr = [c for c in FL if c.startswith('nf4_cr')]
sqrtf = [c for c in FL if 'sqrt' in c]
if __name__ == '__main__':
    print('제외 순간값 %d개, 남긴 변화 피처 %d개(%s), 흐름 %d개' % (len(snap), len(CHG), ', '.join(CHG), len(FL)))
    y, grp = df.label.values, df.subject_id.values
    V = {'S0 순간값 제외(변화 피처 유지)': S0, 'S1 S0+활력 기울기': S0 + vs, 'S2 S1+흐름 전체': S0 + vs + FL,
         'S3 S1+흐름(소변만)': S0 + vs + uo, 'S4 S1+흐름(크레아티닌만)': S0 + vs + cr,
         'S5 S2에서 기존 변화 피처 제거': [c for c in S0 if c not in CHG] + vs + FL,
         'S6 S2에서 절대 소변 구간값 제거(상대·변화만)': S0 + vs + [c for c in FL if c not in uo_abs],
         'S7 S2에서 √ 기울기 제거': S0 + vs + [c for c in FL if c not in sqrtf]}
    res = {}
    for k, f in V.items():
        t0 = time.time(); res[k] = cv(f); N.log('%-34s %3d개 CV AUROC %.4f | %.0f초' % (k, len(f), res[k], time.time() - t0))
        json.dump(res, open(N.OUT + '/flow_cv.json', 'w'), ensure_ascii=False)
        if time.time() - T0 > 140: print('시간 제한으로 중단'); break
