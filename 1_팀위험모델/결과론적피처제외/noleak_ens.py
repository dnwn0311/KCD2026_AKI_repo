# -*- coding: utf-8 -*-
"""10단계: V1(B+활력 기울기) 으로 LightGBM / 로지스틱 / ExtraTrees / 앙상블을 교차검증(3회) 비교하고, 시험 세트에서 최종 확인"""
import json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
src = open('noleak_groups.py').read().split("B = sets['B_정의직결제외']")[0].replace("for k, v in G.items(): print(k, len(v))", "")
exec(src)
B = sets['B_정의직결제외']; V1 = B + G['vit_slope']
y, grp = df.label.values, df.subject_id.values
numc = [c for c in V1 if c not in cats]
def prep(tr, te):
    med = tr[numc].median(); mu = tr[numc].fillna(med).mean(); sd = tr[numc].fillna(med).std().replace(0, 1)
    ind = [c for c in numc if tr[c].isna().mean() > 0.02]
    f = lambda d: pd.concat([((d[numc].fillna(med) - mu) / sd).clip(-5, 5), d[ind].isna().astype(float).add_suffix('_na')], axis=1).values
    return f(tr), f(te)
rk = lambda a: rankdata(a) / len(a)
def models(trd, ted, seed):
    out = {}
    out['LGB'] = np.mean([lgb.LGBMClassifier(random_state=seed * 10 + s, **P).fit(trd[V1], trd.label).predict_proba(ted[V1])[:, 1] for s in range(3)], 0)
    a, b = prep(trd, ted)
    out['LR'] = LogisticRegression(max_iter=3000, C=0.02).fit(a, trd.label).predict_proba(b)[:, 1]
    out['ET'] = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.3, n_jobs=2, random_state=seed).fit(a, trd.label).predict_proba(b)[:, 1]
    return out
t0 = time.time(); res = {}; oofs = {k: np.zeros(len(df)) for k in ('LGB', 'LR', 'ET')}
d = df.reset_index(drop=True); aucs = {k: [] for k in ('LGB', 'LR', 'ET', 'LGB+LR', 'LGB+LR+ET', 'LGB(.7)+LR(.3)')}
for rep in range(2):
    rng = np.random.RandomState(rep); ug = np.unique(grp); perm = dict(zip(ug, rng.permutation(len(ug)))); g2 = np.array([perm[g] for g in grp])
    oo = {k: np.zeros(len(d)) for k in ('LGB', 'LR', 'ET')}
    for tr, va in GroupKFold(5).split(d, y, g2):
        o = models(d.iloc[tr], d.iloc[va], rep)
        for k in o: oo[k][va] = rk(o[k])
    for k in oo: aucs[k].append(roc_auc_score(y, oo[k]))
    aucs['LGB+LR'].append(roc_auc_score(y, oo['LGB'] + oo['LR'])); aucs['LGB+LR+ET'].append(roc_auc_score(y, oo['LGB'] + oo['LR'] + oo['ET']))
    aucs['LGB(.7)+LR(.3)'].append(roc_auc_score(y, .7 * oo['LGB'] + .3 * oo['LR']))
    N.log('반복 %d 완료 %.0f초' % (rep + 1, time.time() - t0))
res['cv'] = {k: round(float(np.mean(v)), 4) for k, v in aucs.items()}
N.log('CV AUROC: ' + ', '.join('%s %.4f' % (k, v) for k, v in res['cv'].items()))
json.dump(res, open(N.OUT + '/ens_cv.json', 'w'), ensure_ascii=False)
