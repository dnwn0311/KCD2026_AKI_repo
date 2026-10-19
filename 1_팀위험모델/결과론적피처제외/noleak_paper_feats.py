# -*- coding: utf-8 -*-
"""Yu et al. 2026 (KRCP, CKD 진행 예측) 의 피처 방식을 AKI 시간 척도로 옮긴 피처
 (1) eGFR 기준치·eGFR slope: CKD-EPI 2021(인종 계수 없음)로 크레아티닌 -> eGFR. 입실 전 기준 eGFR, 창 안 eGFR 기울기(하루당),
     직전 두 측정 간 slope(논문식: (eGFR_t - eGFR_t-1)/간격), 기준 대비 변화율, eGFR<30/<60 구간 표시(논문의 30 미만 급증 비선형)
 (2) 약물 누적 노출 기간: 신독성 약물군별 '기준 시점 전 48시간 안 노출 시간'과 '입원 후 누적 노출 시간', 동시 노출 약물군 수,
     반코마이신+피페라실린-타조박탐 동시 노출 시간, 반코마이신 누적 용량
 기준 시점(입실 48시간) 이전에 시작한 처방만, 노출은 기준 시점까지만 센다(이후 정보 없음). 저장 _work/noleak/paper_feats.parquet"""
import numpy as np, pandas as pd
c = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id', 'hadm_id', 'admittime', 'intime', 'time_zero'])
w = pd.read_parquet('_work/team_fixed/wide.parquet', columns=['stay_id', 'age_at_icu', 'gender', 'creat_baseline_7d', 'creat_last_pre_icu']).set_index('stay_id').reindex(c.stay_id)
ids = c.stay_id.values.astype('int64'); n = len(ids)
age = pd.to_numeric(w.age_at_icu, errors='coerce').values; fem = (w.gender.astype(str).str.upper().str[0] == 'F').values
def egfr(scr, age, fem):
    k = np.where(fem, 0.7, 0.9); a = np.where(fem, -0.241, -0.302); r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(fem, 1.012, 1.0)
F = {'stay_id': ids}
with np.errstate(all='ignore'):
    b1 = pd.to_numeric(w.creat_baseline_7d, errors='coerce').values; b2 = pd.to_numeric(w.creat_last_pre_icu, errors='coerce').values
    base = np.fmin(b1, b2); base = np.where((base > 0.1) & (base < 30), base, np.nan)
    eb = egfr(base, age, fem); F['pp_egfr_baseline'] = eb; F['pp_egfr_base_lt30'] = np.where(np.isnan(eb), np.nan, (eb < 30).astype(float)); F['pp_egfr_base_lt60'] = np.where(np.isnan(eb), np.nan, (eb < 60).astype(float))
    l = pd.read_parquet('_work/noleak/win_labs.parquet'); l = l[(l.code == 5) & (l.value > 0.1) & (l.value < 30)].sort_values(['stay_id', 'hour'])
    pos = pd.Series(np.arange(n), index=ids); i = pos.reindex(l.stay_id).values
    l['e'] = egfr(l.value.values, age[i], fem[i]); l['x'] = l.hour / 24
    g = l.groupby('stay_id')
    def ols(d):
        if len(d) < 2 or d.x.max() - d.x.min() < 0.25: return np.nan
        return np.polyfit(d.x, d.e, 1)[0]
    F['pp_egfr_slope_day'] = g.apply(ols).reindex(ids).values
    F['pp_egfr_slope_day_24h'] = l[l.hour >= 24].groupby('stay_id').apply(ols).reindex(ids).values
    last2 = g.tail(2); gg = last2.groupby('stay_id')
    dd = gg.agg(e0=('e', 'first'), e1=('e', 'last'), x0=('x', 'first'), x1=('x', 'last'), k=('e', 'size'))
    dd['s'] = np.where((dd.k == 2) & (dd.x1 - dd.x0 > 1 / 24), (dd.e1 - dd.e0) / (dd.x1 - dd.x0), np.nan)
    F['pp_egfr_slope_last2'] = dd.s.reindex(ids).values
    elast = g.e.last().reindex(ids).values; efirst = g.e.first().reindex(ids).values
    F['pp_egfr_pct_change_vs_base'] = (elast - eb) / eb * 100
    F['pp_egfr_pct_change_in_window'] = (elast - efirst) / efirst * 100
# ---- 약물 노출
CLS = {'vanc': r'^vancomycin', 'piptazo': r'piperacillin', 'aminogly': r'^(gentamicin|tobramycin sulfate|tobramycin - dosing|amikacin)', 'ampho': r'amphotericin',
       'antiviral': r'^(acyclovir|ganciclovir|valganciclovir|foscarnet)', 'nsaid': r'^(ketorolac|ibuprofen|naproxen|celecoxib)', 'acei_arb': r'^(lisinopril|captopril|enalapril|losartan|valsartan|sacubitril)',
       'cni': r'^(tacrolimus|cyclosporine)', 'polymyxin': r'^(colistin|polymyxin b$|polymyxin b sulfate$)', 'tmp_smx': r'^(sulfameth|sulfamethoxazole)', 'contrast': r'^(iohexol|iodixanol|iopamidol)', 'loop': r'^(furosemide|bumetanide|torsemide)'}
r = pd.read_parquet('_work/rx.parquet', columns=['hadm_id', 'drug', 'start', 'stop', 'dose', 'per24'], filters=[('hadm_id', 'in', list(c.hadm_id))])
r['d'] = r.drug.str.lower(); r = r.merge(c[['hadm_id', 'stay_id', 'admittime', 'time_zero']], on='hadm_id')
r = r[r.start < r.time_zero]
r['stop2'] = r.stop.fillna(r.start + pd.Timedelta(hours=24)).clip(upper=r.time_zero)
H = lambda a, b: ((b - a).dt.total_seconds() / 3600).clip(lower=0)
win0 = r.time_zero - pd.Timedelta(hours=48)
grid_t = None
CL = {}
for k, p in CLS.items():
    s = r[r.d.str.contains(p, regex=True, na=False)]
    if k == 'aminogly': s = s[~s.d.str.contains('inhal|ophth|topical|oint', regex=True)]
    h48 = H(np.maximum(s.start, win0.loc[s.index]), s.stop2).groupby(s.stay_id).sum()
    hall = H(s.start, s.stop2).groupby(s.stay_id).sum()
    F['pp_rx_%s_h48' % k] = h48.reindex(ids).fillna(0).clip(upper=48).values
    F['pp_rx_%s_h_since_adm' % k] = hall.reindex(ids).fillna(0).values
    act = s[(s.start <= s.time_zero) & (s.stop.isna() | (s.stop >= s.time_zero - pd.Timedelta(hours=6)))]   # 기준 시점 근처 활성
    CL[k] = set(act.stay_id)
nephro = [k for k in CLS if k not in ('loop',)]
F['pp_rx_n_nephro_active_t0'] = np.array([sum(s in CL[k] for k in nephro) for s in ids], dtype=float)
F['pp_rx_nephro_burden_h48'] = sum(F['pp_rx_%s_h48' % k] for k in nephro)
F['pp_rx_vanc_piptazo_both_t0'] = np.array([(s in CL['vanc']) and (s in CL['piptazo']) for s in ids], dtype=float)
F['pp_rx_vanc_piptazo_coexp_h48'] = np.minimum(F['pp_rx_vanc_h48'], F['pp_rx_piptazo_h48'])
v = r[r.d.str.contains(r'^vancomycin', regex=True, na=False)]
vd = (v.per24.fillna(0) * H(np.maximum(v.start, win0.loc[v.index]), v.stop2) / 24).groupby(v.stay_id).sum()
F['pp_rx_vanc_cum_dose_48h'] = vd.reindex(ids).fillna(0).values
out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan); out = out.astype({k: 'float32' for k in out.columns if k != 'stay_id'}); out['stay_id'] = out.stay_id.astype('int64')
out.to_parquet('_work/noleak/paper_feats.parquet', index=False)
eg = [k for k in out.columns if k.startswith('pp_egfr')]; rx = [k for k in out.columns if k.startswith('pp_rx')]
print('eGFR 피처 %d개 (결측 %.1f%%), 약물 노출 피처 %d개' % (len(eg), out[eg].isna().mean().mean() * 100, len(rx)), '| 체류', out.stay_id.nunique())
print('노출 비율(%%): ' + ', '.join('%s %.1f' % (k, 100 * (out['pp_rx_%s_h48' % k] > 0).mean()) for k in CLS))
