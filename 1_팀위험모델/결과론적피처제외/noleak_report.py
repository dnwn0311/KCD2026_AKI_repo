# -*- coding: utf-8 -*-
"""11단계: 시험 세트 최종 확인 (학습 7,689건으로만 학습). 집계만 저장: noleak_out/report_*.csv/json, SHAP 요약 그림용 데이터"""
import json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
src = open('noleak_groups.py').read().split("B = sets['B_정의직결제외']")[0].replace("for k, v in G.items(): print(k, len(v))", "")
exec(src)
A = sets['A_전체280']; B = sets['B_정의직결제외']; V1 = B + G['vit_slope']; Cc = sets['C_엄격제외']
sp = {k: df[df.split == k] for k in ('train', 'val', 'test')}
tr, te = sp['train'], sp['test']; ytr, yte = tr.label.values, te.label.values
rk = lambda a: rankdata(a) / len(a)
def lgbm(feats, n=5, d=tr, t=te):
    ms = [lgb.LGBMClassifier(random_state=s, **P).fit(d[feats], d.label) for s in range(n)]
    return np.mean([m.predict_proba(t[feats])[:, 1] for m in ms], 0), ms
def lr(feats, d=tr, t=te):
    num = [c for c in feats if c not in cats]; med = d[num].median(); mu = d[num].fillna(med).mean(); sd = d[num].fillna(med).std().replace(0, 1)
    ind = [c for c in num if d[c].isna().mean() > 0.02]
    f = lambda x: pd.concat([((x[num].fillna(med) - mu) / sd).clip(-5, 5), x[ind].isna().astype(float).add_suffix('_na')], axis=1).values
    return LogisticRegression(max_iter=3000, C=0.02).fit(f(d), d.label).predict_proba(f(t))[:, 1]
R = {}
R['A 전체 280개 (기준, 결과론적 포함)'], _ = lgbm(A)
R['B 정의직결 49개 제외'], _ = lgbm(B)
R['C 엄격 제외(정의직결+합병증·치료반응 106개)'], _ = lgbm(Cc)
p_v1, ms = lgbm(V1); R['V1 B + 활력징후 기울기 (LightGBM)'] = p_v1
p_lr = lr(V1); R['V1 로지스틱'] = p_lr
R['V1 최종: LightGBM 0.7 + 로지스틱 0.3 (순위 평균)'] = 0.7 * rk(p_v1) + 0.3 * rk(p_lr)
rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
def ci(p, f=roc_auc_score):
    v = [f(yte[i], p[i]) for i in boots if yte[i].min() != yte[i].max()]; return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
def topk(p, r):
    k = int(round(r * len(yte))); a = np.zeros(len(yte), bool); a[np.argsort(-p, kind='stable')[:k]] = True; tp = (a & (yte == 1)).sum(); return tp / yte.sum(), tp / k
rows = []
final_key = 'V1 최종: LightGBM 0.7 + 로지스틱 0.3 (순위 평균)'
for k, p in R.items():
    r = {'모델': k, 'AUROC': round(roc_auc_score(yte, p), 3), 'AUROC 95%구간': ci(p), 'AUPRC': round(average_precision_score(yte, p), 3)}
    for q in (0.05, 0.10, 0.20):
        s, ppv = topk(p, q); r['상위%d%% 민감도' % (q * 100)] = round(s, 3); r['상위%d%% 양성예측도' % (q * 100)] = round(ppv, 3)
    rows.append(r)
tab = pd.DataFrame(rows); tab.to_csv(N.OUT + '/report_test.csv', index=False, encoding='utf-8-sig'); print(tab.to_string(index=False))
# 쌍 비교 (같은 부트스트랩)
def diff(a, b):
    v = [roc_auc_score(yte[i], R[a][i]) - roc_auc_score(yte[i], R[b][i]) for i in boots if yte[i].min() != yte[i].max()]
    return round(float(np.mean(v)), 3), [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
ka, kb = list(R)[0], list(R)[1]
print('최종 - B :', diff(final_key, kb)); print('최종 - A :', diff(final_key, ka)); print('V1 - B :', diff(list(R)[3], kb))
# 하위집단: 크레아티닌 기준 양성 / 소변량만 양성
w = pd.read_parquet('_work/team_fixed/wide.parquet', columns=['stay_id', 'aki_by_creat', 'aki_by_uo']).set_index('stay_id').reindex(te.stay_id.values)
neg = yte == 0; bc = w.aki_by_creat.fillna(False).astype(bool).values; bu = w.aki_by_uo.fillna(False).astype(bool).values
sub = {}
for k in (ka, kb, final_key):
    for nm, pos in (('크레아티닌 기준 양성', bc), ('소변량만 양성', bu & ~bc)):
        m = neg | pos; sub[(k, nm)] = round(roc_auc_score(pos[m].astype(int), R[k][m]), 3)
print('하위집단 AUROC:', sub)
# ROC 점
fp, tp, _ = roc_curve(yte, R[final_key]); fa, ta, _ = roc_curve(yte, R[ka]); fb, tb, _ = roc_curve(yte, R[kb])
idx = np.linspace(0, 1, 101)
pd.DataFrame({'FPR': idx, 'TPR_A전체': np.interp(idx, fa, ta), 'TPR_B제외': np.interp(idx, fb, tb), 'TPR_최종': np.interp(idx, fp, tp)}).round(4).to_csv(N.OUT + '/report_roc.csv', index=False)
# 최종 LightGBM의 중요도(평균 |SHAP|, 시험 세트)와 묶음별 합
contrib = np.mean([m.booster_.predict(te[V1], pred_contrib=True)[:, :-1] for m in ms], 0)
imp = pd.Series(np.abs(contrib).mean(0), index=V1)
share = (100 * imp / imp.sum()).sort_values(ascending=False)
def grp_of(c):
    if c in G['vit_slope']: return '활력징후 기울기(새)'
    if c.startswith('atc'): return '약물'
    if c.endswith('_n_hours_obs'): return '측정 횟수(시간 수)'
    if c.startswith(('hr_', 'sbp_', 'dbp_', 'mbp_', 'rr_', 'temp_c_', 'tr_mbp', 'tr_vaso', 'vasopressor')): return '활력징후·승압제'
    if c in ('creat_baseline_7d', 'creat_last_pre_icu', 'esrd_known_at_t0', 'ckd_icd_prior_adm', 'rrt_before_t0'): return '신장 기저(입실 전)'
    if c in ('age_at_icu', 'gender', 'first_careunit', 'admission_type', 'weight_kg'): return '기본 정보'
    if c in ('n_kidney_relevant_classes',): return '약물'
    return '기타'
g = share.groupby(grp_of).sum().sort_values(ascending=False).round(1)
print('묶음별 기여도(%):', g.to_dict()); print('상위 15:', {k: round(v, 1) for k, v in share.head(15).items()})
share.round(3).to_csv(N.OUT + '/report_importance_V1.csv', header=['기여도(%)'], encoding='utf-8-sig')
json.dump({'제외한 정의직결 피처': [c for c in sets['A_전체280'] if c not in B], '추가 활력 기울기 피처': G['vit_slope'], '합병증·치료반응(엄격 제외 추가분)': [c for c in B if c not in Cc]}, open(N.OUT + '/report_feature_lists.json', 'w'), ensure_ascii=False, indent=1)
