# -*- coding: utf-8 -*-
"""RMS(제곱평균제곱근) 계열 흐름 피처: 활력징후 7종 + 시간당 소변량 + 크레아티닌
 - rms        : sqrt(mean(x^2))                       (창 전체 크기)
 - rmssd      : sqrt(mean(diff(x)^2))                 (시간 간 출렁임)
 - rms_resid  : 선형 추세를 뺀 잔차의 RMS               (추세 주변 흔들림)
 - rms_dev    : 나중 24시간이 자기 첫 24시간 평균에서 벗어난 정도의 RMS (기준 대비 이탈)
창: 48시간 전체, 최근 24시간, 최근 12시간.  저장: _work/noleak/rms_feats.parquet (stay_id 는 int64 유지)"""
import numpy as np, pandas as pd
src = open('noleak_flow_build.py').read().split("def slope(M, a, b_):")[0]
exec(src)                     # ids, pos, U(소변량 mL/kg/h 1시간 격자), wt 등
v = pd.read_parquet('_work/noleak/win_vitals.parquet'); v['hb'] = np.minimum(v.hour.astype(int), 47)
G = {'uo': U}
for code, k in ((0, 'hr'), (1, 'sbp'), (2, 'dbp'), (3, 'map'), (4, 'rr'), (5, 'spo2'), (6, 'temp')):
    d_ = v[v.code == code].groupby(['stay_id', 'hb']).value.mean().reset_index()
    M = np.full((len(ids), 48), np.nan); M[pos.reindex(d_.stay_id).values, d_.hb.values] = d_.value.values; G[k] = M
def resid_rms(M, a, b):
    X = M[:, a:b]; x = np.arange(a, b, dtype='float64'); k = ~np.isnan(X); n = k.sum(1)
    xm = np.where(k, x, np.nan); mx = np.nanmean(xm, 1); my = np.nanmean(X, 1)
    sxy = np.nansum((xm - mx[:, None]) * (X - my[:, None]), 1); sxx = np.nansum((xm - mx[:, None]) ** 2, 1)
    b1 = sxy / np.where(sxx > 0, sxx, np.nan); r = X - (my[:, None] + b1[:, None] * (x - mx[:, None]))
    return np.where(n >= 4, np.sqrt(np.nanmean(r ** 2, 1)), np.nan)
F = {'stay_id': ids.astype('int64')}
with np.errstate(all='ignore'):
    for k, M in G.items():
        for a, nm in ((0, '48h'), (24, '24h'), (36, '12h')):
            X = M[:, a:]
            F['rms_%s_rms_%s' % (k, nm)] = np.sqrt(np.nanmean(X ** 2, 1))
            F['rms_%s_rmssd_%s' % (k, nm)] = np.sqrt(np.nanmean(np.diff(X, axis=1) ** 2, 1))
            F['rms_%s_resid_%s' % (k, nm)] = resid_rms(M, a, 48)
        base = np.nanmean(M[:, :24], 1)
        F['rms_%s_dev_vs_day1' % k] = np.sqrt(np.nanmean((M[:, 24:] - base[:, None]) ** 2, 1))
        F['rms_%s_rmssd_ratio_late_early' % k] = np.sqrt(np.nanmean(np.diff(M[:, 24:], axis=1) ** 2, 1)) / np.sqrt(np.nanmean(np.diff(M[:, :24], axis=1) ** 2, 1))
    # 크레아티닌: 불규칙 측정이므로 연속 측정 간 변화(시간당)로 RMSSD
    l = pd.read_parquet('_work/noleak/win_labs.parquet'); l = l[(l.code == 5) & (l.value > 0.1) & (l.value < 30)].sort_values(['stay_id', 'hour'])
    l['dv'] = l.groupby('stay_id').value.diff(); l['dh'] = l.groupby('stay_id').hour.diff(); l['dlog'] = np.log(l.value).groupby(l.stay_id).diff()
    l = l[l.dh > 0.5]
    g = l.groupby('stay_id')
    F['rms_cr_rmssd_per_day'] = (np.sqrt(g.apply(lambda d: np.mean((d.dv / d.dh * 24) ** 2)))).reindex(ids).values
    F['rms_cr_rmssd_log'] = (np.sqrt(g.dlog.apply(lambda x: np.mean(x ** 2)))).reindex(ids).values
    F['rms_cr_rms_rise_only'] = (np.sqrt(g.dv.apply(lambda x: np.mean(np.clip(x, 0, None) ** 2)))).reindex(ids).values
out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
out = out.astype({c: 'float32' for c in out.columns if c != 'stay_id'})
out.to_parquet('_work/noleak/rms_feats.parquet', index=False)
print('RMS 피처', out.shape[1] - 1, '개, 체류', out.stay_id.nunique(), '| 결측률 평균 %.1f%%' % (out.drop(columns='stay_id').isna().mean().mean() * 100))
