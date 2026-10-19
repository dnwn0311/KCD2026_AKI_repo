# -*- coding: utf-8 -*-
"""6단계: 전체 11,000건에서 환자 단위 5-겹 x 3회 교차검증으로 특징 집합 비교 (고정 트리 수, 기본 설정). 집계만 저장.
시험 1,658건(양성 261) 하나로는 AUROC 오차가 ±0.02 라 집합 간 차이를 가르기 어려워 전체 사례로 다시 잰다."""
import json, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
import noleak_common as N
df, sets, cats, l1, l2, nfc = N.load()
y, grp = df.label.values, df.subject_id.values
P = dict(learning_rate=0.03, num_leaves=6, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_estimators=300, verbose=-1, n_jobs=2)
res = json.load(open(N.OUT + '/cv_results.json')) if N.os.path.exists(N.OUT + '/cv_results.json') else {}
oofs = {}
for name in sys.argv[1:]:
    feats = sets[name]; t0 = time.time(); X = df[feats]; aucs, aps = [], []; oof_all = np.zeros((3, len(y)))
    for rep in range(3):
        rng = np.random.RandomState(rep); ug = np.unique(grp); perm = dict(zip(ug, rng.permutation(len(ug))))
        g2 = np.array([perm[g] for g in grp])
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, y, g2):
            m = lgb.LGBMClassifier(random_state=rep, **P).fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y, oof)); aps.append(average_precision_score(y, oof)); oof_all[rep] = oof
    np.save('%s/_oof_%s.npy' % (N.OUT, name.split('_')[0]), oof_all.mean(0))
    res[name] = {'n_features': len(feats), 'cv_auroc_mean': round(float(np.mean(aucs)), 4), 'cv_auroc_sd': round(float(np.std(aucs)), 4), 'cv_auprc_mean': round(float(np.mean(aps)), 4)}
    N.log('%-26s 특징 %3d | CV AUROC %.4f ±%.4f | AUPRC %.4f | %.0f초' % (name, len(feats), np.mean(aucs), np.std(aucs), np.mean(aps), time.time() - t0))
    json.dump(res, open(N.OUT + '/cv_results.json', 'w'), ensure_ascii=False, indent=1)
