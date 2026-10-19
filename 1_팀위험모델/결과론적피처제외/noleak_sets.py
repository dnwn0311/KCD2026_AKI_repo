# -*- coding: utf-8 -*-
"""3단계: 특징 집합별 비교 (같은 분할 split_random, 같은 기본 설정, 검증 세트로 조기 종료) — 집계만 저장"""
import json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import noleak_common as N
df, sets, cats, l1, l2, nfc = N.load()
N.log('사례 %d, 정의직결 제외 %d개, 합병증·치료반응 제외 %d개, 새 특징 %d개' % (len(df), len(l1), len(l2), len(nfc)))
sp = {k: df[df.split == k] for k in ('train', 'val', 'test')}
params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=2.0, verbose=-1, random_state=42, n_jobs=2)
only = sys.argv[1:] 
res = json.load(open(N.OUT + '/sets.json')) if N.os.path.exists(N.OUT + '/sets.json') else {}
for name, feats in sets.items():
    if only and name not in only: continue
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=3000, **params)
    m.fit(sp['train'][feats], sp['train'].label, eval_set=[(sp['val'][feats], sp['val'].label)], eval_metric='auc',
          callbacks=[lgb.early_stopping(150, verbose=False)])
    pv, pt = m.predict_proba(sp['val'][feats])[:, 1], m.predict_proba(sp['test'][feats])[:, 1]
    res[name] = {'n_features': len(feats), 'best_iter': int(m.best_iteration_), 'val_auroc': roc_auc_score(sp['val'].label, pv),
                 'test_auroc': roc_auc_score(sp['test'].label, pt), 'test_auprc': average_precision_score(sp['test'].label, pt)}
    N.log('%-24s 특징 %3d | 검증 %.3f | 시험 AUROC %.3f AUPRC %.3f | 트리 %d | %.0f초' % (
        name, len(feats), res[name]['val_auroc'], res[name]['test_auroc'], res[name]['test_auprc'], m.best_iteration_, time.time() - t0))
    json.dump(res, open(N.OUT + '/sets.json', 'w'), ensure_ascii=False, indent=1)
