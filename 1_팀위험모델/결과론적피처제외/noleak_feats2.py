# -*- coding: utf-8 -*-
"""2-b단계: 입실 전 입원 맥락(입원→입실 시간, 입원 경로, 과거 입원 횟수) 특징을 new_feats.parquet 에 추가"""
import numpy as np, pandas as pd
c = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id', 'subject_id', 'hadm_id', 'admittime', 'intime'])
a = pd.read_parquet('_work/admissions.parquet', columns=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_location'])
f = pd.DataFrame({'stay_id': c.stay_id})
f['nf_pre_hours_admit_to_icu'] = ((c.intime - c.admittime).dt.total_seconds() / 3600).clip(lower=0).values
f['nf_pre_admit_hour'] = c.admittime.dt.hour.values
f['nf_pre_icu_hour'] = c.intime.dt.hour.values
loc = c[['hadm_id']].merge(a[['hadm_id', 'admission_location']].drop_duplicates('hadm_id'), on='hadm_id', how='left')['admission_location']
f['nf_pre_loc_ed'] = loc.fillna('').str.contains('EMERGENCY').astype('float32').values
f['nf_pre_loc_transfer'] = loc.fillna('').str.contains('TRANSFER').astype('float32').values
f['nf_pre_loc_clinic_referral'] = loc.fillna('').str.contains('REFERRAL|CLINIC|PHYSICIAN').astype('float32').values
m = c.merge(a[['subject_id', 'hadm_id', 'admittime', 'dischtime']].rename(columns={'admittime': 'a_adm', 'dischtime': 'a_dis', 'hadm_id': 'h2'}), on='subject_id')
prior = m[m.a_dis < m.admittime]
g = prior.groupby('stay_id')
f = f.set_index('stay_id')
f['nf_pre_n_prior_adm'] = g.size().reindex(f.index).fillna(0)
f['nf_pre_n_prior_adm_365d'] = prior[prior.a_dis >= prior.admittime - pd.Timedelta(days=365)].groupby('stay_id').size().reindex(f.index).fillna(0)
last = (prior.assign(gap=(prior.admittime - prior.a_dis).dt.total_seconds() / 86400).groupby('stay_id').gap.min())
f['nf_pre_days_since_last_disch'] = last.reindex(f.index)
f = f.astype('float32').reset_index()
nf = pd.read_parquet('_work/noleak/new_feats.parquet')
nf = nf[[x for x in nf.columns if not x.startswith('nf_pre_')]].merge(f, on='stay_id', how='left')
nf.to_parquet('_work/noleak/new_feats.parquet', index=False)
print('추가 후 새 특징', nf.shape[1] - 1, '개; 입실 전 맥락', f.shape[1] - 1, '개')
