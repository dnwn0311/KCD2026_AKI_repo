# -*- coding: utf-8 -*-
"""4단계: 학습+검증 구간(9,342건)에서 환자 단위 5-겹 교차검증으로 LightGBM 설정을 무작위 탐색 (시험 세트는 쓰지 않음)
사용: python noleak_tune.py <특징집합 이름> [최대 설정 수]   (150초 안에 끝나도록 이어 실행 가능)"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import noleak_common as N
name = sys.argv[1]; maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 60
df, sets, cats, l1, l2, nfc = N.load()
feats = sets[name]
d = df[df.split != 'test'].reset_index(drop=True)
X, y, grp = d[feats], d.label.values, d.subject_id.values
folds = list(GroupKFold(5).split(X, y, grp))
path = '%s/tune_%s.jsonl' % (N.OUT, name.split('_')[0])
done = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
rng = np.random.RandomState(len(done) + 7)
t0 = time.time()
def cv_auc(p):
    oof = np.zeros(len(y)); its = []
    for tr, va in folds:
        m = lgb.LGBMClassifier(n_estimators=2500, verbose=-1, n_jobs=2, random_state=42, subsample_freq=1, **p)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; its.append(m.best_iteration_)
    return roc_auc_score(y, oof), int(np.median(its))
BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0)
n = 0
cands = [BASE] if not done else []
while len(done) + len(cands) < maxn and time.time() - t0 < 100:
    cands.append(dict(learning_rate=0.03, num_leaves=int(rng.choice([4, 6, 8, 12, 15, 24, 31])),
                      min_child_samples=int(rng.choice([15, 30, 50, 80, 120])), subsample=float(rng.choice([0.6, 0.7, 0.8, 0.9])),
                      colsample_bytree=float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.7])), reg_lambda=float(rng.choice([0, 2, 5, 10, 30])),
                      reg_alpha=float(rng.choice([0, 0.1, 1])), max_depth=int(rng.choice([-1, 3, 4, 6])),
                      min_split_gain=float(rng.choice([0, 0.01, 0.05])), path_smooth=float(rng.choice([0, 5, 20]))))
    break
res = []
with open(path, 'a') as fo:
    for p in cands:
        pass
    # 후보를 시간 안에서 하나씩 평가
    k = 0
    while time.time() - t0 < 150 and len(done) + k < maxn:
        if k < len(cands): p = cands[k]
        else:
            p = dict(learning_rate=0.03, num_leaves=int(rng.choice([4, 6, 8, 12, 15, 24, 31])),
                     min_child_samples=int(rng.choice([15, 30, 50, 80, 120])), subsample=float(rng.choice([0.6, 0.7, 0.8, 0.9])),
                     colsample_bytree=float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.7])), reg_lambda=float(rng.choice([0, 2, 5, 10, 30])),
                     reg_alpha=float(rng.choice([0, 0.1, 1])), max_depth=int(rng.choice([-1, 3, 4, 6])),
                     min_split_gain=float(rng.choice([0, 0.01, 0.05])), path_smooth=float(rng.choice([0, 5, 20])))
        auc, it = cv_auc(p)
        fo.write(json.dumps({'params': p, 'cv_auroc': auc, 'iter': it}) + '\n'); fo.flush()
        N.log('%s #%d CV AUROC %.4f (트리 %d) %s' % (name.split('_')[0], len(done) + k + 1, auc, it, 'leaves=%d mcs=%d col=%.1f lam=%g depth=%d' % (p['num_leaves'], p['min_child_samples'], p['colsample_bytree'], p['reg_lambda'], p.get('max_depth', -1))))
        k += 1
allr = [json.loads(l) for l in open(path)]
b = max(allr, key=lambda r: r['cv_auroc'])
print('설정 %d개 중 최고 CV AUROC %.4f, 기본 설정 %.4f' % (len(allr), b['cv_auroc'], allr[0]['cv_auroc']))
