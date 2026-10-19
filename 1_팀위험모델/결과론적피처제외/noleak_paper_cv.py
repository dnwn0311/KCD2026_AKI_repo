# -*- coding: utf-8 -*-
"""Yu 2026 방식 적용 비교 (교차검증 환자 단위 5-겹 x 3회, LightGBM). 기준 = F2 + RMS(소변·크레아티닌 14개)
 P1 +eGFR 기준치·slope  P2 +약물 누적 노출  P3 둘 다  P4 P3+결측 표시 변수  P5 P3+학습 폴드 중앙값 대체(논문 방식)"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
rm = pd.read_parquet('_work/noleak/rms_feats.parquet'); pp = pd.read_parquet('_work/noleak/paper_feats.parquet')
df = df.merge(rm, on='stay_id', how='left').merge(pp, on='stay_id', how='left'); assert len(df) == 11000
RS = [c for c in rm.columns if c.split('_')[1] in ('uo', 'cr')]
EG = [c for c in pp.columns if c.startswith('pp_egfr')]; RX = [c for c in pp.columns if c.startswith('pp_rx')]
P0 = F2 + RS; P3 = P0 + EG + RX
y, grp = df.label.values, df.subject_id.values
def cv2(feats, mode=None, reps=3):
    a = []
    for rep in range(reps):
        rg = np.random.RandomState(rep); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp]); oof = np.zeros(len(y))
        for t_, v_ in GroupKFold(5).split(df, y, g2):
            Xt, Xv = df.iloc[t_][feats].copy(), df.iloc[v_][feats].copy()
            num = [c for c in feats if c not in cats]
            if mode == 'ind':
                ind = [c for c in num if Xt[c].isna().mean() > 0.02]
                Xt = pd.concat([Xt, Xt[ind].isna().astype('int8').add_suffix('_na')], axis=1); Xv = pd.concat([Xv, Xv[ind].isna().astype('int8').add_suffix('_na')], axis=1)
            if mode == 'median':
                med = Xt[num].median(); Xt[num] = Xt[num].fillna(med); Xv[num] = Xv[num].fillna(med)
            oof[v_] = lgb.LGBMClassifier(random_state=rep, **P).fit(Xt, y[t_]).predict_proba(Xv)[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a)), float(np.std(a))
V = [('P0 기준(F2+RMS14)', P0, None), ('P1 +eGFR(%d)' % len(EG), P0 + EG, None), ('P2 +약물노출(%d)' % len(RX), P0 + RX, None), ('P3 +eGFR+약물노출', P3, None),
     ('P4 P3+결측표시', P3, 'ind'), ('P5 P3+중앙값대체(논문)', P3, 'median')]
sel = sys.argv[1:] or [v[0][:2] for v in V]
path = N.OUT + '/paper_cv.json'; res = json.load(open(path)) if os.path.exists(path) else {}
for nm, f, mo in V:
    if nm[:2] not in sel: continue
    t0 = time.time(); m_, s_ = cv2(f, mo); res[nm] = [round(m_, 4), round(s_, 4)]
    N.log('%-24s %3d개 CV AUROC %.4f ±%.4f | %.0f초' % (nm, len(f), m_, s_, time.time() - t0)); json.dump(res, open(path, 'w'), ensure_ascii=False)
