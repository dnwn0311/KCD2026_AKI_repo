# -*- coding: utf-8 -*-
"""RMS 피처를 F2(순간값 제외+흐름) 위에 묶음별로 더해 교차검증(5-겹 x 3회) 비교. 사용: python noleak_rms_cv.py 0|1 (전반/후반)"""
import json, os, re, sys, time, warnings
import numpy as np, pandas as pd
exec(open('noleak_flow_cv2.py').read().split("y, grp = df.label")[0])
rm = pd.read_parquet('_work/noleak/rms_feats.parquet'); df = df.merge(rm, on='stay_id', how='left'); assert len(df) == 11000
R_ = [c for c in rm.columns if c != 'stay_id']
VIT = ('hr', 'sbp', 'dbp', 'map', 'rr', 'spo2', 'temp')
grp_ = lambda c: c.split('_')[1]
Rv = [c for c in R_ if grp_(c) in VIT]; Ru = [c for c in R_ if grp_(c) == 'uo']; Rc = [c for c in R_ if grp_(c) == 'cr']
kind = lambda c: 'rmssd' if 'rmssd' in c else ('resid' if '_resid_' in c else ('dev' if 'dev_vs' in c else 'rms'))
V = [('F2 기준', F2), ('F2 + RMS 전체(%d)' % len(R_), F2 + R_), ('F2 + RMS 활력(%d)' % len(Rv), F2 + Rv), ('F2 + RMS 소변(%d)' % len(Ru), F2 + Ru), ('F2 + RMS 크레아티닌(%d)' % len(Rc), F2 + Rc)]
for k in ('rms', 'rmssd', 'resid', 'dev'):
    f = [c for c in R_ if kind(c) == k]; V.append(('F2 + %s만(%d)' % (k, len(f)), F2 + f))
half = int(sys.argv[1]) if len(sys.argv) > 1 else 0
V = V[:5] if half == 0 else V[5:]
y, grp = df.label.values, df.subject_id.values
path = N.OUT + '/rms_cv.json'; res = json.load(open(path)) if os.path.exists(path) else {}
for nm, f in V:
    t0 = time.time(); res[nm] = cv(f); N.log('%-26s %3d개 CV AUROC %.4f | %.0f초' % (nm, len(f), res[nm], time.time() - t0)); json.dump(res, open(path, 'w'), ensure_ascii=False)
