# -*- coding: utf-8 -*-
"""2-c단계: 활력징후를 1시간 평균 격자(48칸)로 만들어 여러 창의 기울기·일차 차이·RMSSD(연속 차이 제곱평균의 제곱근)·변동계수를 추가
저장: _work/noleak/new_feats3.parquet (nf3_*)"""
import numpy as np, pandas as pd
VIT = ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temp_c']
ids = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id'])['stay_id'].values
pos = pd.Series(np.arange(len(ids)), index=ids)
v = pd.read_parquet('_work/noleak/win_vitals.parquet')
v['hb'] = np.minimum(v.hour.astype(int), 47)
T = 48
def slope(M, a, b):   # 열 a..b-1 의 선형 기울기 (단위: 값/시간 x 24 = 하루)
    S = M[:, a:b]; x = np.arange(a, b, dtype='float64'); ok = ~np.isnan(S)
    n = ok.sum(1); sx = (ok * x).sum(1); sy = np.nansum(S, 1); sxx = (ok * x * x).sum(1); sxy = np.nansum(S * x, 1)
    den = n * sxx - sx * sx
    with np.errstate(all='ignore'):
        r = (n * sxy - sx * sy) / np.where(den > 1e-9, den, np.nan) * 24
    return np.where(n >= 3, r, np.nan)
out = {'stay_id': ids}
for i, k in enumerate(VIT):
    d = v[v.code == i].groupby(['stay_id', 'hb']).value.mean().reset_index()
    M = np.full((len(ids), T), np.nan); M[pos.reindex(d.stay_id).values, d.hb.values] = d.value.values
    for a, nm in ((42, '6h'), (36, '12h'), (24, '24h'), (0, '48h')):
        out['nf3_%s_slope_%s' % (k, nm)] = slope(M, a, T)
    with np.errstate(all='ignore'):
        d1, d2 = np.nanmean(M[:, :24], 1), np.nanmean(M[:, 24:], 1)
        out['nf3_%s_day2_minus_day1' % k] = d2 - d1
        # RMSSD: 인접한 두 시간 평균이 모두 있는 칸의 차이
        df_ = np.diff(M, axis=1); out['nf3_%s_rmssd' % k] = np.sqrt(np.nanmean(df_ ** 2, 1))
        out['nf3_%s_cv' % k] = np.nanstd(M, 1) / np.nanmean(M, 1)
        out['nf3_%s_max_drop_6h' % k] = np.nanmin(np.array([np.nanmean(M[:, j + 6:j + 12], 1) - np.nanmean(M[:, j:j + 6], 1) for j in range(0, 37, 6)]), 0)
F = pd.DataFrame(out); F = F.astype({c: 'float32' for c in F.columns if c != 'stay_id'}); F['stay_id'] = F['stay_id'].astype('int64')
F.to_parquet('_work/noleak/new_feats3.parquet', index=False)
print('nf3', F.shape[1] - 1, '개; 결측률 평균 %.1f%%' % (F.drop(columns='stay_id').isna().mean().mean() * 100))
