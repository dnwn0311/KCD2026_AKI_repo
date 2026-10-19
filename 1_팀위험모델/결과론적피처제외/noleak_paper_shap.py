# -*- coding: utf-8 -*-
"""논문 방식(Yu 2026: eGFR·약물 노출 피처 + XGBoost·LightGBM·CatBoost·RF soft voting) vs 우리 방식(LightGBM, F2+RMS14) SHAP 비교
학습: 학습+검증 9,342건 / SHAP: 시험 1,658건. 환자 단위 SHAP 값은 PC(_work/noleak/shap_paper/)에만 저장.
사용: python noleak_paper_shap.py boost | rf | summary"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
rm = pd.read_parquet('_work/noleak/rms_feats.parquet'); pp = pd.read_parquet('_work/noleak/paper_feats.parquet')
df = df.merge(rm, on='stay_id', how='left').merge(pp, on='stay_id', how='left'); assert len(df) == 11000
RS = [c for c in rm.columns if c.split('_')[1] in ('uo', 'cr')]; EG = [c for c in pp.columns if c.startswith('pp_egfr')]; RX = [c for c in pp.columns if c.startswith('pp_rx')]
P0 = F2 + RS; P3 = P0 + EG + RX
D = '_work/noleak/shap_paper'; os.makedirs(D, exist_ok=True)
trn = df[df.split != 'test']; te = df[df.split == 'test']; yt = trn.label.values
def to_num(X):
    X = X.copy()
    for c in cats:
        if c in X: X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
    return X
st = sys.argv[1]
if st == 'boost':
    t0 = time.time()
    ms = [lgb.LGBMClassifier(random_state=s, **P).fit(trn[P0], yt) for s in range(3)]
    np.save(D + '/ours_lgb.npy', np.mean([m.booster_.predict(te[P0], pred_contrib=True) for m in ms], 0))
    ms = [lgb.LGBMClassifier(random_state=s, **P).fit(trn[P3], yt) for s in range(3)]
    np.save(D + '/paper_lgb.npy', np.mean([m.booster_.predict(te[P3], pred_contrib=True) for m in ms], 0))
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=5, n_jobs=2, tree_method='hist', eval_metric='logloss', random_state=0).fit(to_num(trn[P3]), yt)
    np.save(D + '/paper_xgb.npy', m.get_booster().predict(xgb.DMatrix(to_num(te[P3])), pred_contribs=True))
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(iterations=300, learning_rate=0.08, depth=5, l2_leaf_reg=5, verbose=0, thread_count=2, random_seed=0).fit(to_num(trn[P3]), yt)
    np.save(D + '/paper_cat.npy', m.get_feature_importance(Pool(to_num(te[P3])), type='ShapValues'))
    print('boost 완료 %.0f초' % (time.time() - t0))
elif st == 'rf':
    import shap
    from sklearn.ensemble import RandomForestClassifier
    a, b = to_num(trn[P3]), to_num(te[P3]); med = a.median(); a, b = a.fillna(med), b.fillna(med)
    t0 = time.time(); rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=15, max_features='sqrt', n_jobs=2, random_state=0).fit(a, yt)
    sv = shap.TreeExplainer(rf).shap_values(b, check_additivity=False)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    np.save(D + '/paper_rf.npy', sv); print('rf SHAP(시험 전체) %.0f초' % (time.time() - t0))
else:
    Xte = to_num(te[P3])
    def share(v, names):
        s = pd.Series(np.abs(v).mean(0), index=names); return 100 * s / s.sum()
    S = {}
    for k in ('lgb', 'xgb', 'cat'): S[k] = share(np.load('%s/paper_%s.npy' % (D, k))[:, :-1], P3)
    if os.path.exists(D + '/paper_rf.npy'): S['rf'] = share(np.load(D + '/paper_rf.npy'), P3)
    ours = share(np.load(D + '/ours_lgb.npy')[:, :-1], P0)
    ens = pd.concat(S, axis=1).mean(1)          # 모델별 기여도(%)의 평균 = soft voting 동일 가중치
    boost = np.mean([np.load('%s/paper_%s.npy' % (D, k)) for k in ('lgb', 'xgb', 'cat')], 0)   # 같은 로그오즈 단위 3종 평균
    np.save(D + '/paper_boost_avg.npy', boost)
    vals = boost[:, :-1]; rows = []
    for j, c in enumerate(P3):
        x = Xte[c].values.astype(float); s = vals[:, j]; ok = ~np.isnan(x)
        if ok.sum() > 30 and np.nanstd(x) > 0:
            q1, q3 = np.nanquantile(x, [0.25, 0.75]); rho = spearmanr(x[ok], s[ok])[0]; hi = s[ok & (x >= q3)].mean(); lo = s[ok & (x <= q1)].mean()
        else: rho = hi = lo = np.nan
        rows.append({'특징': c, '논문방식_앙상블_기여도(%)': ens[c], 'LightGBM(%)': S['lgb'][c], 'XGBoost(%)': S['xgb'][c], 'CatBoost(%)': S['cat'][c], 'RF(%)': S['rf'][c] if 'rf' in S else np.nan,
                     '우리방식_LGB(%)': ours.get(c, np.nan), '값-SHAP 스피어만': rho, '값 상위25% 평균SHAP': hi, '값 하위25% 평균SHAP': lo, '결측률(%)': 100 * np.isnan(x).mean(),
                     '새 논문 피처': c in EG + RX})
    t = pd.DataFrame(rows).sort_values('논문방식_앙상블_기여도(%)', ascending=False); t.insert(0, '순위', np.arange(1, len(t) + 1))
    t['우리방식_순위'] = t['우리방식_LGB(%)'].rank(ascending=False, method='min')
    t.round(4).to_csv(N.OUT + '/paper_shap_compare.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(t.head(30)[['순위', '특징', '논문방식_앙상블_기여도(%)', 'LightGBM(%)', 'XGBoost(%)', 'CatBoost(%)', 'RF(%)', '우리방식_LGB(%)', '우리방식_순위', '값-SHAP 스피어만', '값 상위25% 평균SHAP', '값 하위25% 평균SHAP', '결측률(%)']].round(3).to_string(index=False))
    print('새 논문 피처(eGFR+약물) 합계: 앙상블 %.1f%% (eGFR %.1f, 약물 %.1f), 순위 최상위 %d' % (t[t['새 논문 피처']]['논문방식_앙상블_기여도(%)'].sum(), ens[EG].sum(), ens[RX].sum(), t[t['새 논문 피처']]['순위'].min()))
    print('모델별 상위10 겹침:', {k: len(set(S[k].nlargest(10).index) & set(ens.nlargest(10).index)) for k in S})
    print('논문 피처 상위:', t[t['새 논문 피처']].head(8)[['순위', '특징', '논문방식_앙상블_기여도(%)', '값-SHAP 스피어만']].round(3).to_string(index=False))
