# -*- coding: utf-8 -*-
"""7단계: 새 특징 묶음별로 B(정의직결 제외)에 하나씩 더해 교차검증 AUROC 변화를 본다 (묶음 선택은 CV 기준)"""
import json, sys, time, re, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
import noleak_common as N
df, sets, cats, l1, l2, nfc = N.load()
nf = [c for c in nfc if not c.endswith('_sqrt')]
VIT = ('heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temp_c')
def vit(c): return c.startswith('nf_') and c.split('_')[1] in ('heart', 'sbp', 'dbp', 'map', 'resp', 'spo2', 'temp') 
def stat(c): return c.rsplit('_', 1)[-1] if not c.endswith('slope_day') else 'slope_day'
G = {
 'vit_slope': [c for c in nf if vit(c) and re.search(r'(slope_day|slope12_day|_d6|last_mean6)$', c)],
 'vit_stat':  [c for c in nf if vit(c) and not re.search(r'(slope_day|slope12_day|_d6|last_mean6)$', c)],
 'lab_slope': [c for c in nf if c.split('_')[1] in ('glucose', 'calcium', 'hemoglobin', 'wbc', 'platelet', 'lactate', 'inr') and re.search(r'(slope_day|slope12_day|_d6|last_mean6)$', c)],
 'lab_stat':  [c for c in nf if c.split('_')[1] in ('glucose', 'calcium', 'hemoglobin', 'wbc', 'platelet', 'lactate', 'inr') and not re.search(r'(slope_day|slope12_day|_d6|last_mean6)$', c)],
 'burden':    [c for c in nf if c.startswith(('nf_h_', 'nf_frac_', 'nf_shock', 'nf_pulse'))],
 'pre':       [c for c in nf if c.startswith('nf_pre_')],
}
for k, v in G.items(): print(k, len(v))
y, grp = df.label.values, df.subject_id.values
P = dict(learning_rate=0.03, num_leaves=6, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_estimators=300, verbose=-1, n_jobs=2)
def cv(feats, reps=3):
    X = df[feats]; a = []
    for rep in range(reps):
        rng = np.random.RandomState(rep); ug = np.unique(grp); perm = dict(zip(ug, rng.permutation(len(ug)))); g2 = np.array([perm[g] for g in grp]); oof = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, y, g2):
            oof[va] = lgb.LGBMClassifier(random_state=rep, **P).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a))
B = sets['B_정의직결제외']; out = {}
base = cv(B); out['B'] = base; N.log('B 기준 %.4f' % base)
for k in sys.argv[1:]:
    t0 = time.time(); out[k] = cv(B + G[k]); N.log('B + %-10s (%3d개) CV AUROC %.4f (%+.4f) %.0f초' % (k, len(G[k]), out[k], out[k] - base, time.time() - t0))
json.dump(out, open(N.OUT + '/groups_%s.json' % '_'.join(sys.argv[1:]), 'w'), ensure_ascii=False)
