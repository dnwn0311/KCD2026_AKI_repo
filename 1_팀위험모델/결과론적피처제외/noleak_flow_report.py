# -*- coding: utf-8 -*-
"""순간값 제외 + 흐름 피처 최종 시험 평가 (학습 7,689건으로만 학습, 시험 1,658건). 집계만 저장."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
exec(open('noleak_flow_cv2.py').read().split("res = {}")[0])
sp = {k: df[df.split == k] for k in ('train', 'val', 'test')}; tr, te = sp['train'], sp['test']; yte = te.label.values
def lg(feats, n=5):
    ms = [lgb.LGBMClassifier(random_state=s, **P).fit(tr[feats], tr.label) for s in range(n)]; return np.mean([m.predict_proba(te[feats])[:, 1] for m in ms], 0), ms
R = {}
R['A 전체 280개 (기준)'], _ = lg(A)
R['이전 최선: 정의직결 49개 제외+활력기울기+앙상블'] = None
V1 = [c for c in sets['B_정의직결제외']] + vs
pv, _ = lg(V1); R['이전 최선: 정의직결 49개 제외+활력기울기+앙상블'] = .7 * rk(pv) + .3 * rk(lrp(V1, tr, te))
pl, ms2 = lg(F2); R['F2 순수 추세 LightGBM'] = pl
R['F2 순수 추세 (LGB .7 + LR .3)'] = .7 * rk(pl) + .3 * rk(lrp(F2, tr, te))
pl1, _ = lg(F1); R['F1 순간값 제외+변화·흐름 (LGB .7 + LR .3)'] = .7 * rk(pl1) + .3 * rk(lrp(F1, tr, te))
rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
def ci(p): v = [roc_auc_score(yte[i], p[i]) for i in boots if yte[i].min() != yte[i].max()]; return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
def topk(p, r): k = int(round(r * len(yte))); a = np.zeros(len(yte), bool); a[np.argsort(-p, kind='stable')[:k]] = True; tp = (a & (yte == 1)).sum(); return round(tp / yte.sum(), 3), round(tp / k, 3)
rows = []
for k, p in R.items():
    r = {'모델': k, 'AUROC': round(roc_auc_score(yte, p), 3), '95%구간': ci(p), 'AUPRC': round(average_precision_score(yte, p), 3)}
    for q in (0.05, 0.10, 0.20): r['상위%d%% 민감도/양성예측도' % (q * 100)] = topk(p, q)
    rows.append(r)
tab = pd.DataFrame(rows); print(tab.to_string(index=False)); tab.to_csv(N.OUT + '/flow_report_test.csv', index=False, encoding='utf-8-sig')
ks = list(R); fin = ks[3]
def diff(a, b): v = [roc_auc_score(yte[i], R[a][i]) - roc_auc_score(yte[i], R[b][i]) for i in boots if yte[i].min() != yte[i].max()]; return round(float(np.mean(v)), 3), [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
print('F2 앙상블 - 기준 A:', diff(fin, ks[0]), '| - 이전 최선:', diff(fin, ks[1]))
w = pd.read_parquet('_work/team_fixed/wide.parquet', columns=['stay_id', 'aki_by_creat', 'aki_by_uo']).set_index('stay_id').reindex(te.stay_id.values)
neg = yte == 0; bc = w.aki_by_creat.fillna(False).astype(bool).values; bu = w.aki_by_uo.fillna(False).astype(bool).values
sub = {}
for k in (ks[0], ks[1], fin):
    for nm, pos in (('크레아티닌 기준 양성', bc), ('소변량만 양성', bu & ~bc)):
        m = neg | pos; sub['%s | %s' % (k[:14], nm)] = round(roc_auc_score(pos[m].astype(int), R[k][m]), 3)
print('하위집단 AUROC:', sub)
fp, tp, _ = roc_curve(yte, R[fin]); fa, ta, _ = roc_curve(yte, R[ks[0]]); fb, tb, _ = roc_curve(yte, R[ks[1]]); ix = np.linspace(0, 1, 101)
pd.DataFrame({'FPR': ix, 'TPR_A전체': np.interp(ix, fa, ta), 'TPR_이전최선': np.interp(ix, fb, tb), 'TPR_F2': np.interp(ix, fp, tp)}).round(4).to_csv(N.OUT + '/flow_report_roc.csv', index=False)
contrib = np.mean([m.booster_.predict(te[F2], pred_contrib=True)[:, :-1] for m in ms2], 0); imp = pd.Series(np.abs(contrib).mean(0), index=F2); share = (100 * imp / imp.sum()).sort_values(ascending=False)
def grp_of(c):
    if c.startswith('nf4_uo') or c in ('tr_uo_trend_12h',): return '소변량 흐름·추세'
    if c.startswith('nf4_cr') or c.startswith('tr_cr_') or c.startswith('tr_bun'): return '크레아티닌 흐름·추세'
    if c in vs: return '활력징후 기울기'
    if c.startswith('atc') or c == 'n_kidney_relevant_classes': return '약물'
    if c.endswith('_n_hours_obs'): return '측정 시간 수'
    if c.startswith(('hr_', 'sbp_', 'dbp_', 'mbp_', 'rr_', 'temp_c_', 'tr_mbp', 'tr_vaso', 'vasopressor')): return '활력징후·승압제'
    if c in ('age_at_icu', 'gender', 'first_careunit', 'admission_type', 'weight_kg'): return '기본 정보'
    if c in ('creat_baseline_7d', 'creat_last_pre_icu', 'esrd_known_at_t0', 'ckd_icd_prior_adm', 'rrt_before_t0'): return '신장 기저(입실 전)'
    return '기타(전해질·수분·투석 등)'
g = share.groupby(grp_of).sum().sort_values(ascending=False).round(1); print('묶음별 기여도(%):', g.to_dict())
print('상위 15:', {k: round(v, 1) for k, v in share.head(15).items()}); share.round(3).to_csv(N.OUT + '/flow_importance_F2.csv', header=['기여도(%)'], encoding='utf-8-sig')
json.dump({'제외한 순간값': snap, '남긴 변화 피처(기존)': [c for c in CHG if c not in REF], '새 흐름 피처': FL, '순수 추세에서 뺀 기준치 대비': REF}, open(N.OUT + '/flow_feature_lists.json', 'w'), ensure_ascii=False, indent=1)
