# -*- coding: utf-8 -*-
"""[플로우 피처] 크레아티닌·소변량을 '한 순간의 값'이 아니라 시간에 따른 흐름(궤적)으로 만든다.
 - 소변량: 측정 간격에 나눠 담은 1시간 격자(mL/kg/h) -> 6시간 구간 8개의 흐름, 구간 간 변화, √ 변환 기울기
 - 크레아티닌: 시간 순서 측정값 -> 첫 값 대비 비(ratio) 궤적, 기울기(원값·log·√), 변화량, 기저치 대비 변화
저장: _work/noleak/flow_feats.parquet (nf4_*).  환자 행은 출력하지 않는다."""
import time, numpy as np, pandas as pd
t0 = time.time()
c = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id', 'hadm_id', 'intime', 'time_zero'])
w = pd.read_parquet('_work/team_fixed/wide.parquet', columns=['stay_id', 'weight_kg', 'creat_baseline_7d', 'creat_last_pre_icu']).set_index('stay_id')
ids = c.stay_id.values; pos = pd.Series(np.arange(len(ids)), index=ids); n = len(ids); T = 48
wt = pd.to_numeric(w.reindex(ids).weight_kg, errors='coerce').values; wt = np.where((wt > 20) & (wt < 300), wt, np.nanmedian(wt))
cm = c.set_index('stay_id')
# ---- 소변량 1시간 격자
o = pd.read_parquet('_work/kidney_output.parquet'); o = o[o.urine & o.stay_id.isin(set(ids))]
o = o.join(cm[['intime', 'time_zero']], on='stay_id'); o = o[(o.tav <= o.time_zero) & (o.ct < o.time_zero) & (o.ct > o.intime - pd.Timedelta(hours=24))]
o = o.groupby(['stay_id', 'ct'], as_index=False).agg(tav=('tav', 'max'), vol=('vol', 'sum'), intime=('intime', 'first')).sort_values(['stay_id', 'ct'])
gap = o.groupby('stay_id').ct.diff().dt.total_seconds().values / 3600; gap = np.where(np.isnan(gap), 1.0, np.clip(gap, 0.25, 24))
end = ((o.ct - o.intime).dt.total_seconds() / 3600).values; st = end - gap
S = np.zeros((n, T)); C = np.zeros((n, T))   # 담은 양, 덮은 시간
si = pos.reindex(o.stay_id).values
lo = np.floor(np.maximum(st, 0)).astype(int); hi = np.minimum(np.floor(end), T - 1).astype(int); ok = (end > 0) & (lo <= hi)
rep = (hi - lo + 1)[ok]; idx = np.repeat(np.where(ok)[0], rep)
off = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep); b = lo[idx] + off
ov = np.clip(np.minimum(end[idx], b + 1) - np.maximum(st[idx], b), 0, None)
np.add.at(S, (si[idx], b), o.vol.values[idx] * ov / gap[idx]); np.add.at(C, (si[idx], b), ov)
U = np.where(C >= 0.5, S / np.maximum(C, 1e-9) / wt[:, None], np.nan)   # mL/kg/h
print('소변량 격자: 체류', int((~np.isnan(U)).any(1).sum()), '칸 채움률 %.1f%%' % (100 * (~np.isnan(U)).mean()), '%.0f초' % (time.time() - t0))
def slope(M, a, b_):
    X = M[:, a:b_]; x = np.arange(a, b_, dtype='float64'); k = ~np.isnan(X); m = k.sum(1); sx = (k * x).sum(1); sy = np.nansum(X, 1); sxx = (k * x * x).sum(1); sxy = np.nansum(X * x, 1)
    den = m * sxx - sx * sx
    with np.errstate(all='ignore'): r = (m * sxy - sx * sy) / np.where(den > 1e-9, den, np.nan) * 24
    return np.where(m >= 4, r, np.nan)
F = {'stay_id': ids}
with np.errstate(all='ignore'):
    B6 = np.stack([np.nanmean(U[:, 6 * k:6 * k + 6], 1) for k in range(8)], 1)
    for k in range(8): F['nf4_uo_bin%d' % k] = B6[:, k]                                     # 흐름: 6시간 구간 소변량 8개
    ref = np.nanmean(B6[:, :4], 1)
    for k in range(4, 8): F['nf4_uo_rel%d' % k] = B6[:, k] / np.where(ref > 0.02, ref, np.nan)     # 자기 첫날 대비 상대 흐름
    for k in range(1, 8): F['nf4_uo_dbin%d' % k] = B6[:, k] - B6[:, k - 1]                         # 구간 간 변화
    Us = np.sqrt(np.clip(U, 0, None))
    for a, nm in ((36, '12h'), (24, '24h'), (0, '48h')):
        F['nf4_uo_slope_%s' % nm] = slope(U, a, T)
        F['nf4_uo_sqrt_slope_%s' % nm] = slope(Us, a, T)                                          # 제곱근 변환 후 기울기
    F['nf4_uo_log_ratio_12h'] = np.log((np.nanmean(U[:, 36:], 1) + 0.05) / (np.nanmean(U[:, 24:36], 1) + 0.05))
    F['nf4_uo_log_ratio_24_24'] = np.log((np.nanmean(U[:, 24:], 1) + 0.05) / (np.nanmean(U[:, :24], 1) + 0.05))
    d = np.diff(B6, axis=1); F['nf4_uo_max_drop_bin'] = np.nanmin(d, 1); F['nf4_uo_n_decline_bins'] = (d < -0.05).sum(1).astype(float)
    F['nf4_uo_sqrt_rmssd'] = np.sqrt(np.nanmean(np.diff(Us, axis=1) ** 2, 1))
    F['nf4_uo_peak_to_last'] = np.nanmax(B6, 1) - B6[:, 7]
# ---- 크레아티닌 궤적
l = pd.read_parquet('_work/noleak/win_labs.parquet'); l = l[l.code == 5]; l = l[(l.value > 0.1) & (l.value < 30)].sort_values(['stay_id', 'hour'])
cr_i = pos.reindex(l.stay_id).values; h = np.minimum(l.hour.values, 47.99)
CR = np.full((n, T), np.nan)
for hb in range(T):
    pass
d_ = pd.DataFrame({'i': cr_i, 'hb': np.floor(h).astype(int), 'v': l.value.values}).groupby(['i', 'hb']).v.mean().reset_index()
CR[d_.i.values, d_.hb.values] = d_.v.values
base = pd.to_numeric(w.reindex(ids).creat_baseline_7d, errors='coerce').values; pre = pd.to_numeric(w.reindex(ids).creat_last_pre_icu, errors='coerce').values
with np.errstate(all='ignore'):
    first_idx = np.argmax(~np.isnan(CR), 1); has = (~np.isnan(CR)).any(1)
    first = np.where(has, CR[np.arange(n), first_idx], np.nan)
    ff = pd.DataFrame(CR.T).ffill().values.T; last = ff[:, -1]
    nobs = (~np.isnan(CR)).sum(1); F['nf4_cr_n_obs'] = nobs.astype(float)
    for hh in (12, 24, 36, 47): F['nf4_cr_ratio_first_h%d' % hh] = ff[:, hh] / first      # 첫 값 대비 비 궤적 (순간 수준이 아니라 변화)
    F['nf4_cr_delta_last_first'] = last - first; F['nf4_cr_ratio_last_first'] = last / first
    F['nf4_cr_log_ratio_last_first'] = np.log(last / first); F['nf4_cr_sqrt_ratio_last_first'] = np.sqrt(last / first)
    refb = np.fmin(np.where(np.isnan(base), pre, base), np.where(np.isnan(pre), base, pre)); refb = np.where(refb > 0, refb, np.nan)
    F['nf4_cr_ratio_to_baseline'] = last / refb; F['nf4_cr_delta_to_baseline'] = last - refb
    F['nf4_cr_first_ratio_to_baseline'] = first / refb
    F['nf4_cr_slope_all'] = slope(CR, 0, T); F['nf4_cr_slope_24h'] = slope(np.where(np.arange(T) >= 24, CR, np.nan), 0, T)
    F['nf4_cr_slope_log'] = slope(np.log(CR), 0, T); F['nf4_cr_slope_sqrt'] = slope(np.sqrt(CR), 0, T)
    F['nf4_cr_slope_log_24h'] = slope(np.where(np.arange(T) >= 24, np.log(CR), np.nan), 0, T)
    run = np.fmin.accumulate(np.where(np.isnan(CR), np.inf, CR), axis=1); prev = np.concatenate([np.full((n, 1), np.inf), run[:, :-1]], 1)
    rise = np.where(~np.isnan(CR) & np.isfinite(prev), CR - prev, np.nan); F['nf4_cr_max_rise'] = np.nanmax(rise, 1)
    F['nf4_cr_max_ratio_from_runmin'] = np.nanmax(np.where(~np.isnan(CR) & np.isfinite(prev), CR / prev, np.nan), 1)
    F['nf4_cr_rise_24h'] = last - ff[:, 23]
    F['nf4_cr_up_0_3'] = (np.nan_to_num(last - np.fmin(first, np.nanmin(CR, 1))) >= 0.3).astype(float)
out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan); out = out.astype({c: 'float32' for c in out.columns if c != 'stay_id'}); out['stay_id'] = out['stay_id'].astype('int64')
out.to_parquet('_work/noleak/flow_feats.parquet', index=False)
print('플로우 피처', out.shape[1] - 1, '개 | 소변 관련', sum(x.startswith('nf4_uo') for x in out.columns), ', 크레아티닌 관련', sum(x.startswith('nf4_cr') for x in out.columns), '| %.0f초' % (time.time() - t0))
print('결측률 평균 소변 %.1f%% 크레아티닌 %.1f%%' % (out.filter(like='nf4_uo').isna().mean().mean() * 100, out.filter(like='nf4_cr').isna().mean().mean() * 100))
