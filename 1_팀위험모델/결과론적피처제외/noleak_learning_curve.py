# -*- coding: utf-8 -*-
"""학습 곡선: 같은 검증 접힘에서 학습 데이터를 25/50/75/100%만 써서 AUROC가 데이터 양에 얼마나 반응하는지 확인 (F2 순수 추세, 전체 피처 A 비교)"""
import time, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
exec(open('noleak_flow_cv2.py').read().split("y, grp = df.label")[0])
y, grp = df.label.values, df.subject_id.values; d = df.reset_index(drop=True)
for nm, feats in (('F2 순수 추세', F2), ('A 전체 280개', A)):
    row = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        oof_a = []
        for rep in range(2):
            rg = np.random.RandomState(rep); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp]); oof = np.zeros(len(d))
            for t_, v_ in GroupKFold(5).split(d, y, g2):
                sub = np.random.RandomState(rep + 10).choice(t_, int(len(t_) * frac), replace=False)
                oof[v_] = lgb.LGBMClassifier(random_state=rep, **P).fit(d.iloc[sub][feats], y[sub]).predict_proba(d.iloc[v_][feats])[:, 1]
            oof_a.append(roc_auc_score(y, oof))
        row.append('%d%%(n≈%d): %.3f' % (frac * 100, int(len(d) * 0.8 * frac), np.mean(oof_a)))
    print(nm, ' | '.join(row), flush=True)
