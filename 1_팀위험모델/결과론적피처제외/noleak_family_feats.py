# -*- coding: utf-8 -*-
"""켜고 끌 수 있는 세 피처 묶음 (같은 신호·같은 창으로 통일)
 신호 9개: 심박·수축기·이완기·평균동맥압·호흡·SpO2·체온(1시간 평균), 시간당 소변량(mL/kg/h), 크레아티닌(첫 측정 대비 비)
 창 3개: 48시간 전체, 최근 24시간, 최근 12시간
 SLOPE: 선형 기울기(하루당)  |  MEAN: 평균  |  RMS: 제곱평균제곱근(값) + RMSSD(시간 간 변화의 RMS)
 크레아티닌은 순간 수준이 들어가지 않도록 '첫 측정 대비 비' 궤적으로 계산.
저장 _work/noleak/fam_feats.parquet (stay_id int64)"""
import numpy as np, pandas as pd
exec(open('noleak_flow_build.py').read().split("def slope(M, a, b_):")[0])   # ids, pos, U
v = pd.read_parquet('_work/noleak/win_vitals.parquet'); v['hb'] = np.minimum(v.hour.astype(int), 47)
G = {}
for code, k in ((0, 'hr'), (1, 'sbp'), (2, 'dbp'), (3, 'map'), (4, 'rr'), (5, 'spo2'), (6, 'temp')):
    d_ = v[v.code == code].groupby(['stay_id', 'hb']).value.mean().reset_index()
    M = np.full((len(ids), 48), np.nan); M[pos.reindex(d_.stay_id).values, d_.hb.values] = d_.value.values; G[k] = M
G['uo'] = U
l = pd.read_parquet('_work/noleak/win_labs.parquet'); l = l[(l.code == 5) & (l.value > 0.1) & (l.value < 30)]
d_ = l.assign(hb=np.minimum(l.hour.astype(int), 47)).groupby(['stay_id', 'hb']).value.mean().reset_index()
C_ = np.full((len(ids), 48), np.nan); C_[pos.reindex(d_.stay_id).values, d_.hb.values] = d_.value.values
fi = np.argmax(~np.isnan(C_), 1); has = (~np.isnan(C_)).any(1); first = np.where(has, C_[np.arange(len(ids)), fi], np.nan)
G['cr'] = C_ / first[:, None]                                     # 관측 칸만 (기울기용)
CRff = pd.DataFrame(G['cr'].T).ffill().values.T                     # 평균·RMS용 (첫 측정 이후 앞값 채움)
def slope(M, a, b, mn):
    X = M[:, a:b]; x = np.arange(a, b, dtype='float64'); k = ~np.isnan(X); m = k.sum(1); sx = (k * x).sum(1); sy = np.nansum(X, 1); sxx = (k * x * x).sum(1); sxy = np.nansum(X * x, 1); den = m * sxx - sx * sx
    with np.errstate(all='ignore'): r = (m * sxy - sx * sy) / np.where(den > 1e-9, den, np.nan) * 24
    return np.where(m >= mn, r, np.nan)
F = {'stay_id': ids.astype('int64')}
with np.errstate(all='ignore'):
    for k, M in G.items():
        MM = CRff if k == 'cr' else M
        for a, w in ((0, '48h'), (24, '24h'), (36, '12h')):
            F['fam_slope_%s_%s' % (k, w)] = slope(M, a, 48, 2 if k == 'cr' else 3)
            X = MM[:, a:]
            F['fam_mean_%s_%s' % (k, w)] = np.nanmean(X, 1)
            F['fam_rms_%s_%s' % (k, w)] = np.sqrt(np.nanmean(X ** 2, 1))
            F['fam_rms_%s_rmssd_%s' % (k, w)] = np.sqrt(np.nanmean(np.diff(X, axis=1) ** 2, 1))
out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan); out = out.astype({c: 'float32' for c in out.columns if c != 'stay_id'})
out.to_parquet('_work/noleak/fam_feats.parquet', index=False)
for fam in ('slope', 'mean', 'rms'):
    cs = [c for c in out.columns if c.startswith('fam_%s_' % fam)]; print(fam, len(cs), '개, 결측률 %.1f%%' % (out[cs].isna().mean().mean() * 100))
print('체류', out.stay_id.nunique())
