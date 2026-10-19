# -*- coding: utf-8 -*-
import time as _t; T0=_t.time()
"""선행연구 기법 적용: (1) 구간별 기울기 변화(PSC, JCM 2026) (2) 임상 방향 단조 제약 (JCM 2026) (3) 폴드 내 SHAP 순위 기반 특징 축소 (JCM 2026)
교차검증(환자 단위 5-겹 x 3회) 비교. 집계만 출력."""
import json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
exec(open('noleak_flow_cv2.py').read().split("y, grp = df.label")[0])
y, grp = df.label.values, df.subject_id.values
# --- (1) PSC: 처음 24시간 기울기 vs 나중 24시간 기울기의 차 (MAP·심박·호흡·SpO2)
ids = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id']).stay_id.values; pos = pd.Series(np.arange(len(ids)), index=ids)
v = pd.read_parquet('_work/noleak/win_vitals.parquet'); v['hb'] = np.minimum(v.hour.astype(int), 47)
def slope(M, a, b):
    X = M[:, a:b]; x = np.arange(a, b, dtype='float64'); k = ~np.isnan(X); m = k.sum(1); sx = (k * x).sum(1); sy = np.nansum(X, 1); sxx = (k * x * x).sum(1); sxy = np.nansum(X * x, 1); den = m * sxx - sx * sx
    with np.errstate(all='ignore'): r = (m * sxy - sx * sy) / np.where(den > 1e-9, den, np.nan) * 24
    return np.where(m >= 3, r, np.nan)
P_ = {'stay_id': ids}
for code, k in ((3, 'map'), (0, 'hr'), (4, 'rr'), (5, 'spo2'), (1, 'sbp'), (6, 'temp')):
    d_ = v[v.code == code].groupby(['stay_id', 'hb']).value.mean().reset_index(); M = np.full((len(ids), 48), np.nan); M[pos.reindex(d_.stay_id).values, d_.hb.values] = d_.value.values
    s1, s2 = slope(M, 0, 24), slope(M, 24, 48); P_['psc_%s_s1' % k] = s1; P_['psc_%s_s2' % k] = s2; P_['psc_%s' % k] = s2 - s1
    s3 = slope(M, 36, 48); P_['psc_%s_late' % k] = s3 - s2
psc = pd.DataFrame(P_); psc = psc.astype({c: 'float32' for c in psc.columns if c != 'stay_id'}); psc['stay_id'] = psc['stay_id'].astype('int64'); df = df.merge(psc, on='stay_id', how='left'); PSC = [c for c in psc.columns if c != 'stay_id']
F2 = F2
# --- (2) 단조 제약 (+1: 값이 클수록 위험 상승, -1: 값이 클수록 위험 하락)
UP = ['atc_J01G_aminoglycoside', 'atc_J01XA_vancomycin', 'drug_pip_tazo', 'atc_J01XB_polymyxin', 'atc_J02AA_amphotericin', 'atc_J05AB_nucleoside_antiviral', 'atc_M01A_nsaid', 'atc_L04AD_calcineurin', 'n_kidney_relevant_classes', 'vasopressor_hours',
      'tr_cr_delta_24h', 'tr_cr_rise_from_min', 'tr_cr_ratio_min', 'tr_cr_slope_day', 'tr_cr_max_rise', 'tr_cr_rise_vs_ref', 'tr_cr_ratio_baseline', 'tr_bun_delta_24h', 'tr_mbp_lt65_h_24h', 'tr_vaso_h_12h',
      'nf4_cr_slope_all', 'nf4_cr_slope_24h', 'nf4_cr_slope_log', 'nf4_cr_slope_sqrt', 'nf4_cr_slope_log_24h', 'nf4_cr_delta_last_first', 'nf4_cr_ratio_last_first', 'nf4_cr_max_rise', 'nf4_cr_rise_24h']
DOWN = ['tr_uo_trend_12h', 'nf4_uo_slope_12h', 'nf4_uo_slope_24h', 'nf4_uo_slope_48h', 'nf4_uo_sqrt_slope_12h', 'nf4_uo_sqrt_slope_24h', 'nf4_uo_sqrt_slope_48h', 'nf4_uo_log_ratio_12h', 'nf4_uo_log_ratio_24_24']
def mono(feats): return [1 if c in UP else (-1 if c in DOWN else 0) for c in feats]
def cv_run(feats, monotone=False, select=None, reps=3):
    X = df[feats]; a = []
    for rep in range(reps):
        rg = np.random.RandomState(rep); ug = np.unique(grp); pm = dict(zip(ug, rg.permutation(len(ug)))); g2 = np.array([pm[g] for g in grp]); oof = np.zeros(len(y))
        for t_, v_ in GroupKFold(5).split(X, y, g2):
            f = feats
            if select:   # 폴드 학습 데이터에서만 SHAP 순위 산출
                m0 = lgb.LGBMClassifier(random_state=rep, **P).fit(X.iloc[t_], y[t_]); sv = np.abs(shap.TreeExplainer(m0).shap_values(X.iloc[t_].sample(min(3000, len(t_)), random_state=0))); sv = sv[1] if isinstance(sv, list) else sv
                f = list(pd.Series(sv.mean(0), index=feats).sort_values(ascending=False).index[:select])
            Q = dict(P)
            if monotone: Q['monotone_constraints'] = mono(f); Q['monotone_constraints_method'] = 'advanced'
            oof[v_] = lgb.LGBMClassifier(random_state=rep, **Q).fit(df.iloc[t_][f], y[t_]).predict_proba(df.iloc[v_][f])[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a))
res = {}
V = [('L0 F2 (기준)', F2, False, None), ('L1 F2 + PSC', F2 + PSC, False, None), ('L2 F2 + 단조 제약', F2, True, None), ('L3 F2 + PSC + 단조 제약', F2 + PSC, True, None),
     ('L4 F2 + SHAP 상위 60 (폴드 내)', F2, False, 60), ('L5 F2 + SHAP 상위 120 (폴드 내)', F2, False, 120)]
print('단조 제약 대상: 위험↑ %d개, 위험↓ %d개 (F2 안에서 %d / %d)' % (len(UP), len(DOWN), sum(c in F2 for c in UP), sum(c in F2 for c in DOWN)))
for nm, f, mo, sel in V:
    t0 = time.time(); res[nm] = cv_run(f, mo, sel); N.log('%-34s %3d개 CV AUROC %.4f | %.0f초' % (nm, len(f) if not sel else sel, res[nm], time.time() - t0))
    json.dump(res, open(N.OUT + '/lit_cv.json', 'w'), ensure_ascii=False)
    if time.time() - T0 > 145: print('시간 제한 중단'); break
