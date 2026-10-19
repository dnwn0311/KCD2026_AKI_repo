# -*- coding: utf-8 -*-
"""순수 추세 변형: 마지막 값을 기준치와 비교한 피처(현재 수준이 섞임)까지 빼고 창 안 궤적만 남기기 + 앙상블 교차검증"""
import json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
exec(open('noleak_flow_cv.py').read().split("if __name__")[0])
REF = ['tr_cr_ratio_baseline', 'tr_cr_rise_vs_ref', 'nf4_cr_ratio_to_baseline', 'nf4_cr_delta_to_baseline', 'nf4_cr_first_ratio_to_baseline']
F1 = S0 + vs + FL
F2 = [c for c in F1 if c not in REF]
y, grp = df.label.values, df.subject_id.values; d = df.reset_index(drop=True)
rk = lambda a: rankdata(a) / len(a)
def lrp(feats, tr, te):
    num = [c for c in feats if c not in cats]; med = tr[num].median(); mu = tr[num].fillna(med).mean(); sd = tr[num].fillna(med).std().replace(0, 1); ind = [c for c in num if tr[c].isna().mean() > 0.02]
    f = lambda x: pd.concat([((x[num].fillna(med) - mu) / sd).clip(-5, 5), x[ind].isna().astype(float).add_suffix('_na')], axis=1).values
    return LogisticRegression(max_iter=3000, C=0.02).fit(f(tr), tr.label).predict_proba(f(te))[:, 1]
def cv_ens(feats, reps=2):
    out = {'LGB': [], 'LR': [], '앙상블 .7/.3': [], '앙상블 .5/.5': []}
    for rep in range(reps):
        rg = np.random.RandomState(rep); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp]); a = np.zeros(len(d)); b = np.zeros(len(d))
        for t_, v_ in GroupKFold(5).split(d, y, g2):
            tr, te = d.iloc[t_], d.iloc[v_]
            a[v_] = rk(np.mean([lgb.LGBMClassifier(random_state=s, **P).fit(tr[feats], tr.label).predict_proba(te[feats])[:, 1] for s in range(3)], 0)); b[v_] = rk(lrp(feats, tr, te))
        out['LGB'].append(roc_auc_score(y, a)); out['LR'].append(roc_auc_score(y, b)); out['앙상블 .7/.3'].append(roc_auc_score(y, .7 * a + .3 * b)); out['앙상블 .5/.5'].append(roc_auc_score(y, .5 * a + .5 * b))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}
res = {}
for nm, f in (('F2 순수 추세(기준치 대비 제거)', F2), ('F1 순간값 제외+변화·흐름', F1)):
    t0 = time.time(); res[nm] = cv_ens(f); N.log('%-26s %3d개 %s | %.0f초' % (nm, len(f), res[nm], time.time() - t0))
    json.dump(res, open(N.OUT + '/flow_cv2.json', 'w'), ensure_ascii=False)
