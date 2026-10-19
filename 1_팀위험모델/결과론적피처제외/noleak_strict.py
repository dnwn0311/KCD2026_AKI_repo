# -*- coding: utf-8 -*-
"""12단계: 엄격 제외(정의직결+합병증·치료반응) + 활력 기울기 — 같은 최종 방식(LightGBM 0.7 + 로지스틱 0.3)"""
import json, warnings
import numpy as np, pandas as pd
src = open('noleak_report.py').read().split("R = {}")[0]
exec(src)
V1c = Cc + G['vit_slope']
R = {}
pl, _ = lgbm(V1c); pr = lr(V1c)
R['엄격+기울기 최종 (LGB .7 + LR .3)'] = .7 * rk(pl) + .3 * rk(pr); R['엄격+기울기 LightGBM'] = pl
R['엄격 제외만 (C)'], _ = lgbm(Cc)
R['V1 최종 (정의직결 제외+기울기)'] = .7 * rk(lgbm(V1)[0]) + .3 * rk(lr(V1))
rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
def ci(p):
    v = [roc_auc_score(yte[i], p[i]) for i in boots if yte[i].min() != yte[i].max()]; return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
for k, p in R.items(): print('%-34s AUROC %.3f %s AUPRC %.3f' % (k, roc_auc_score(yte, p), ci(p), average_precision_score(yte, p)))
# 교차검증(3회) 참고
from sklearn.model_selection import GroupKFold
y, grp = df.label.values, df.subject_id.values
dd = df.reset_index(drop=True)
for nm, f in (('엄격+기울기', V1c), ('V1', V1)):
    a = []
    for rep in range(2):
        rg = np.random.RandomState(rep); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp]); oo = np.zeros(len(dd)); ol = np.zeros(len(dd))
        for t_, v_ in GroupKFold(5).split(dd, y, g2):
            d_, t2 = dd.iloc[t_], dd.iloc[v_]
            oo[v_] = rk(np.mean([lgb.LGBMClassifier(random_state=s, **P).fit(d_[f], d_.label).predict_proba(t2[f])[:, 1] for s in range(2)], 0)); ol[v_] = rk(lr(f, d_, t2))
        a.append(roc_auc_score(y, .7 * oo + .3 * ol))
    print('교차검증(2회) %s 앙상블 AUROC %.4f' % (nm, np.mean(a)))
pd.DataFrame([{'모델': k, 'AUROC': round(roc_auc_score(yte, p), 3), 'AUROC 95%구간': ci(p), 'AUPRC': round(average_precision_score(yte, p), 3)} for k, p in R.items()]).to_csv(N.OUT + '/report_strict.csv', index=False, encoding='utf-8-sig')
