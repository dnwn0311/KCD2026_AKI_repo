# -*- coding: utf-8 -*-
"""Yu 2026 의 soft-voting 앙상블(XGBoost·LightGBM·CatBoost·RF)을 같은 피처(F2+RMS14+eGFR)로 재현.
사용: python noleak_vote.py <모델> ...  (lgb xgb cat rf lr) -> 교차검증 OOF(1회, 5-겹)와 시험 예측을 _work/noleak/vote/ 에 저장
      python noleak_vote.py combine      -> 조합별 AUROC (확률 평균 = soft voting, 순위 평균 비교)"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
rm = pd.read_parquet('_work/noleak/rms_feats.parquet'); pp = pd.read_parquet('_work/noleak/paper_feats.parquet')
df = df.merge(rm, on='stay_id', how='left').merge(pp, on='stay_id', how='left'); assert len(df) == 11000
RS = [c for c in rm.columns if c.split('_')[1] in ('uo', 'cr')]; EG = [c for c in pp.columns if c.startswith('pp_egfr')]
FE = F2 + RS + EG
D = '_work/noleak/vote'
if os.environ.get('FESET') == 'A':
    FE = A; D = '_work/noleak/vote_A'; os.makedirs(D, exist_ok=True)
dd = df.reset_index(drop=True); y, grp = dd.label.values, dd.subject_id.values
spl = {k: dd[dd.split == k] for k in ('train', 'test')}; trn = dd[dd.split != 'test']   # 시험용 모델은 학습+검증으로 학습 (모든 모델 동일)
num = [c for c in FE if c not in cats]
def to_num(X, ref):
    X = X.copy()
    for c in cats:
        if c in X: X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
    return X
def fit_pred(name, tr, te):
    Xt, Xv, yt = tr[FE], te[FE], tr.label.values
    if name == 'lgb':
        return np.mean([lgb.LGBMClassifier(random_state=s, **P).fit(Xt, yt).predict_proba(Xv)[:, 1] for s in range(3)], 0)
    if name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=5, n_jobs=2, tree_method='hist', eval_metric='logloss', random_state=0)
        return m.fit(to_num(Xt, tr), yt).predict_proba(to_num(Xv, tr))[:, 1]
    if name == 'cat':
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, learning_rate=0.08, depth=5, l2_leaf_reg=5, verbose=0, thread_count=2, random_seed=0)
        return m.fit(to_num(Xt, tr), yt).predict_proba(to_num(Xv, tr))[:, 1]
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        a, b = to_num(Xt, tr), to_num(Xv, tr); med = a.median(); a, b = a.fillna(med), b.fillna(med)
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=15, max_features='sqrt', n_jobs=2, random_state=0).fit(a, yt).predict_proba(b)[:, 1]
    if name == 'lr':
        return lrp(FE, tr, te)
args = sys.argv[1:]
if args and args[0] != 'combine':
    rg = np.random.RandomState(0); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp])
    folds = list(GroupKFold(5).split(dd, y, g2))
    for name in args:
        t0 = time.time(); oof = np.zeros(len(dd))
        for t_, v_ in folds: oof[v_] = fit_pred(name, dd.iloc[t_], dd.iloc[v_])
        pt = fit_pred(name, trn, spl['test'])
        np.save('%s/oof_%s.npy' % (D, name), oof); np.save('%s/test_%s.npy' % (D, name), pt)
        N.log('%s 교차검증 AUROC %.4f | 시험 AUROC %.3f | %.0f초' % (name, roc_auc_score(y, oof), roc_auc_score(spl['test'].label, pt), time.time() - t0))
else:
    from scipy.stats import rankdata
    rk = lambda a: rankdata(a) / len(a)
    O = {m: np.load('%s/oof_%s.npy' % (D, m)) for m in ('lgb', 'xgb', 'cat', 'rf', 'lr') if os.path.exists('%s/oof_%s.npy' % (D, m))}
    T = {m: np.load('%s/test_%s.npy' % (D, m)) for m in O}; yt = spl['test'].label.values
    combos = [('lgb',), ('xgb',), ('cat',), ('rf',), ('lr',), ('lgb', 'lr'), ('lgb', 'xgb', 'cat', 'rf'), ('lgb', 'xgb', 'cat'), ('lgb', 'cat'), ('lgb', 'xgb', 'cat', 'rf', 'lr'), ('lgb', 'cat', 'lr'), ('lgb', 'xgb', 'cat', 'lr')]
    rng = np.random.RandomState(0); boots = [rng.randint(0, len(yt), len(yt)) for _ in range(1000)]
    rows = []
    for cb in combos:
        if not all(m in O for m in cb): continue
        for how in ('확률평균(soft voting)', '순위평균'):
            f = (lambda d: np.mean([d[m] for m in cb], 0)) if how.startswith('확률') else (lambda d: np.mean([rk(d[m]) for m in cb], 0))
            o, t = f(O), f(T); bv = [roc_auc_score(yt[i], t[i]) for i in boots if yt[i].min() != yt[i].max()]
            rows.append({'조합': '+'.join(cb), '결합': how, '교차검증 AUROC': round(roc_auc_score(y, o), 4), '시험 AUROC': round(roc_auc_score(yt, t), 3),
                         '95%구간': [round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)], '시험 AUPRC': round(average_precision_score(yt, t), 3)})
            if len(cb) == 1: break
    t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(N.OUT + ('/vote_results_A.csv' if os.environ.get('FESET') == 'A' else '/vote_results.csv'), index=False, encoding='utf-8-sig')
