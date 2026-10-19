# -*- coding: utf-8 -*-
"""2단계: 창 원자료에서 '원인 쪽' 생리 지표의 기울기·변동·부담 시간 특징을 만든다 (집계 특징만 저장).
출력 _work/noleak/new_feats.parquet  (stay_id + nf_* 열)
"""
import numpy as np, pandas as pd
VIT = ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temp_c']
LABS = ['potassium', 'sodium', 'chloride', 'bicarbonate', 'bun', 'creatinine', 'glucose', 'magnesium', 'phosphate',
        'calcium', 'hemoglobin', 'wbc', 'platelet', 'lactate', 'inr']
USE_LAB = ['glucose', 'calcium', 'hemoglobin', 'wbc', 'platelet', 'lactate', 'inr']   # 신장 결과가 아닌 지표만
ids = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id'])['stay_id'].values
F = pd.DataFrame({'stay_id': ids}).set_index('stay_id')

def per_var(d, name):
    d = d.sort_values(['stay_id', 'hour'])
    g = d.groupby('stay_id')
    out = pd.DataFrame({'n': g.size(), 'mean': g.value.mean(), 'min': g.value.min(), 'max': g.value.max(),
                        'first': g.value.first(), 'last': g.value.last(), 'std': g.value.std()})
    def slope(dd):
        x = dd.hour / 24.0
        gg = dd.assign(x=x, xx=x * x, xy=x * dd.value, y=dd.value).groupby('stay_id')
        n, sx, sy, sxx, sxy = gg.size(), gg.x.sum(), gg.y.sum(), gg.xx.sum(), gg.xy.sum()
        den = n * sxx - sx * sx
        s = (n * sxy - sx * sy) / den.where(den > 1e-9)
        return s.where(n >= 3)
    out['slope_day'] = slope(d)
    out['slope12_day'] = slope(d[d.hour >= 36])
    m_last = d[d.hour >= 42].groupby('stay_id').value.mean()
    m_prev = d[(d.hour >= 24) & (d.hour < 42)].groupby('stay_id').value.mean()
    out['d6'] = m_last - m_prev
    out['last_mean6'] = m_last
    return out.add_prefix('nf_%s_' % name)

v = pd.read_parquet('_work/noleak/win_vitals.parquet')
l = pd.read_parquet('_work/noleak/win_labs.parquet')
parts = []
for i, k in enumerate(VIT):
    parts.append(per_var(v[v.code == i], k))
for i, k in enumerate(LABS):
    if k in USE_LAB:
        parts.append(per_var(l[l.code == i], k))
F = F.join(pd.concat(parts, axis=1))

# 부담 시간(시간 단위로 한 번이라도 기준을 넘은 시간 수)
def hours_where(code, cond, name):
    d = v[v.code == code]
    d = d[cond(d.value)]
    F['nf_%s' % name] = d.assign(hb=d.hour.astype(int)).groupby('stay_id').hb.nunique().reindex(F.index).fillna(0)
hours_where(3, lambda x: x < 65, 'h_map_lt65')
hours_where(3, lambda x: x < 55, 'h_map_lt55')
hours_where(1, lambda x: x < 90, 'h_sbp_lt90')
hours_where(0, lambda x: x > 110, 'h_hr_gt110')
hours_where(4, lambda x: x > 24, 'h_rr_gt24')
hours_where(5, lambda x: x < 92, 'h_spo2_lt92')
hours_where(6, lambda x: x > 38.3, 'h_temp_gt38_3')
# 관찰 시간 수 대비 비율 (측정 빈도가 많다고 위험하게 보이지 않게)
for k in ['map_lt65', 'sbp_lt90', 'hr_gt110', 'rr_gt24']:
    base = {'map_lt65': 'map', 'sbp_lt90': 'sbp', 'hr_gt110': 'heart_rate', 'rr_gt24': 'resp_rate'}[k]
    F['nf_frac_%s' % k] = F['nf_h_%s' % k] / F['nf_%s_n' % base].replace(0, np.nan)
F['nf_shock_index_last'] = F['nf_heart_rate_last'] / F['nf_sbp_last']
F['nf_pulse_pressure_mean'] = F['nf_sbp_mean'] - F['nf_dbp_mean']
# 제곱근 변환 (비음수·치우친 지표)
for k in ['lactate', 'inr', 'wbc', 'glucose', 'platelet']:
    for s in ['max', 'mean', 'last']:
        F['nf_%s_%s_sqrt' % (k, s)] = np.sqrt(F['nf_%s_%s' % (k, s)].clip(lower=0))
for k in ['h_map_lt65', 'h_hr_gt110', 'h_rr_gt24']:
    F['nf_%s_sqrt' % k] = np.sqrt(F['nf_%s' % k])
F = F.astype('float32').reset_index()
F.to_parquet('_work/noleak/new_feats.parquet', index=False)
print('새 특징', F.shape[1] - 1, '개, 체류', len(F))
print('결측률 상위 8:', (F.isna().mean().sort_values(ascending=False).head(8) * 100).round(1).to_dict())
