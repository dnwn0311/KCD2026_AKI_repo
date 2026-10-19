# -*- coding: utf-8 -*-
import time as _t; T0=_t.time()
"""9단계: 여러 창 기울기·RMSSD 묶음(nf3)을 V1(B+활력 기울기)에 더했을 때 교차검증 AUROC 변화"""
import json, re, sys, time, warnings
import numpy as np, pandas as pd
src = open('noleak_groups.py').read().split("B = sets['B_정의직결제외']")[0].replace("for k, v in G.items(): print(k, len(v))", "")
exec(src)
n3 = pd.read_parquet('_work/noleak/new_feats3.parquet'); df = df.merge(n3, on='stay_id', how='left')
n3c = [c for c in n3.columns if c != 'stay_id']
def kind(c): return c.split('_', 2)[-1] if False else re.sub(r'^nf3_[a-z0-9_]+?_(slope_6h|slope_12h|slope_24h|slope_48h|day2_minus_day1|rmssd|cv|max_drop_6h)$', r'\1', c)
KG = {t: [c for c in n3c if kind(c) == t] for t in ('slope_6h', 'slope_12h', 'slope_24h', 'slope_48h', 'day2_minus_day1', 'rmssd', 'cv', 'max_drop_6h')}
B = sets['B_정의직결제외']; vs = G['vit_slope']
y, grp = df.label.values, df.subject_id.values
res = {}
V = {'V1 (B+기울기)': B + vs, 'V1+nf3 전체': B + vs + n3c, 'B+nf3 전체(기존 기울기 없이)': B + n3c}
for k, f in KG.items(): V['V1+%s' % k] = B + vs + f
for k, f in V.items():
    t0 = time.time(); res[k] = cv(f); N.log('%-30s %3d개 CV AUROC %.4f | %.0f초' % (k, len(f), res[k], time.time() - t0))
    json.dump(res, open(N.OUT + '/slopes_cv.json', 'w'), ensure_ascii=False)
    if time.time() - T0 > 150: break
