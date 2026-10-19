# -*- coding: utf-8 -*-
"""8단계: B + 활력징후 기울기에서 약물군(atc3) 등 잡음 후보를 덜어낸 변형 비교 (교차검증 기준)"""
import json, re, sys, time, warnings
import numpy as np
exec(open('noleak_groups.py').read().split("B = sets['B_정의직결제외']")[0].replace("for k, v in G.items(): print(k, len(v))", ""))
B = sets['B_정의직결제외']; vs = G['vit_slope']
atc3 = [c for c in B if c.startswith('atc3_')]
nobs = [c for c in B if c.endswith('_n_hours_obs')]
outl = {}
V = {'V1 B+기울기': B + vs,
     'V2 V1+입실전맥락': B + vs + G['pre'],
     'V3 V1-atc3(%d)' % len(atc3): [c for c in B if c not in atc3] + vs,
     'V4 V3-측정시간수(%d)' % len(nobs): [c for c in B if c not in atc3 and c not in nobs] + vs,
     'V5 V3+입실전맥락': [c for c in B if c not in atc3] + vs + G['pre']}
for k, f in V.items():
    t0 = time.time(); outl[k] = cv(f); N.log('%-26s %3d개 CV AUROC %.4f | %.0f초' % (k, len(f), outl[k], time.time() - t0))
json.dump(outl, open(N.OUT + '/prune.json', 'w'), ensure_ascii=False)
