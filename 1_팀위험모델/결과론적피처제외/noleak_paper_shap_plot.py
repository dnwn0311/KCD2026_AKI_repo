# -*- coding: utf-8 -*-
"""SHAP 비교 그림(PC 전용, 환자 단위 점) + 집계: 병동별 평균 SHAP, 피처 묶음별 기여도"""
import os, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from matplotlib import font_manager as fm
exec(open('noleak_paper_shap.py').read().split("st = sys.argv[1]")[0])
for f in fm.findSystemFonts():
    if 'NotoSansCJK' in f: fm.fontManager.addfont(f)
kf = [n for n in {f.name for f in fm.fontManager.ttflist} if 'Noto Sans CJK' in n]
if kf: plt.rcParams['font.family'] = sorted(kf)[0]
plt.rcParams['axes.unicode_minus'] = False
OUTD = N.OUT + '/paper_shap'; os.makedirs(OUTD, exist_ok=True)
def bee(v, feats, title, fn, note):
    data = te[feats].copy()
    for c in cats:
        if c in data: data[c] = np.nan
    ex = shap.Explanation(values=v[:, :-1], base_values=v[:, -1], data=data.astype(float).values, feature_names=feats)
    plt.figure(figsize=(10.5, 10.8)); shap.plots.beeswarm(ex, max_display=20, show=False, color_bar=True, plot_size=None)
    plt.gca().set_xlabel('SHAP value  (← AKI 예측 점수 낮춤 | 높임 →, 로그오즈)', fontsize=10); plt.title(title, fontsize=11.5)
    plt.gcf().text(0.01, 0.005, note, fontsize=8, color='#555'); plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUTD + '/' + fn, dpi=150); plt.close()
B = np.load(D + '/paper_boost_avg.npy'); O = np.load(D + '/ours_lgb.npy')
bee(B, P3, '논문 방식 (Yu 2026) SHAP summary plot — eGFR·약물 노출 피처 + 앙상블\n(LightGBM·XGBoost·CatBoost SHAP 평균, 시험 %d건, 특징 %d개)' % (len(te), len(P3)), 'shap_beeswarm_논문방식_앙상블.png',
    'RF는 확률 단위 SHAP이라 그림 평균에서 제외(기여도 표에는 포함). 빨강 = 값 높음, 파랑 = 값 낮음, 회색 = 결측·범주형. 인과 아님.')
bee(O, P0, '우리 방식 SHAP summary plot — 순간값 제외 + 흐름·RMS (LightGBM 3회 평균)\n(시험 %d건, 특징 %d개)' % (len(te), len(P0)), 'shap_beeswarm_우리방식_LGB.png', '빨강 = 값 높음, 파랑 = 값 낮음, 회색 = 결측·범주형. 인과 아님.')
# 병동별 평균 SHAP (범주형이라 스피어만 대신)
j = P3.index('first_careunit'); cu = te['first_careunit'].astype(str).values
t = pd.DataFrame({'병동': cu, 'shap': B[:, j]}).groupby('병동').shap.agg(['size', 'mean']).rename(columns={'size': '시험 건수', 'mean': '평균 SHAP(로그오즈)'})
t['시험 건수'] = t['시험 건수'].where(t['시험 건수'] >= 10, np.nan)   # 10건 미만 가림
t = t.dropna().sort_values('평균 SHAP(로그오즈)', ascending=False).round(3); t.to_csv(OUTD + '/careunit_shap.csv', encoding='utf-8-sig'); print(t.to_string())
# 묶음별 기여도 (논문 방식 앙상블 표 기준)
c = pd.read_csv(N.OUT + '/paper_shap_compare.csv')
def grp(f):
    if f.startswith('pp_egfr'): return 'eGFR 기준치·slope (논문 신규)'
    if f.startswith('pp_rx'): return '약물 누적 노출 (논문 신규)'
    if f.startswith(('nf4_uo', 'rms_uo', 'tr_uo')): return '소변량 흐름·변동'
    if f.startswith(('nf4_cr', 'rms_cr', 'tr_cr', 'tr_bun')): return '크레아티닌 변화'
    if f.startswith('nf_') and any(k in f for k in ('heart', 'sbp', 'dbp', 'map', 'resp', 'spo2', 'temp')): return '활력징후 기울기·변화'
    if f.startswith(('atc', 'drug_')) or f == 'n_kidney_relevant_classes': return '약물 표시(기존)'
    if f.endswith('_n_hours_obs'): return '측정 횟수'
    if f.startswith(('hr_', 'sbp_', 'dbp_', 'mbp_', 'rr_', 'temp_c_', 'tr_mbp', 'tr_vaso', 'vasopressor')): return '활력징후·승압제 수준'
    if f in ('age_at_icu', 'gender', 'first_careunit', 'admission_type', 'weight_kg'): return '기본 정보'
    if f in ('creat_baseline_7d', 'creat_last_pre_icu', 'esrd_known_at_t0', 'ckd_icd_prior_adm', 'rrt_before_t0'): return '신장 기저(입실 전)'
    if f.startswith('nf_pre_'): return '입실 전 맥락'
    if f.startswith(('nf_', )): return '기타 검사(젖산·혈당 등)'
    return '전해질·수분·투석 등'
c['묶음'] = c['특징'].map(grp)
g = c.groupby('묶음')[['논문방식_앙상블_기여도(%)', '우리방식_LGB(%)']].sum().sort_values('논문방식_앙상블_기여도(%)', ascending=False).round(1)
g.to_csv(OUTD + '/group_share.csv', encoding='utf-8-sig'); print(g.to_string())
c.to_csv(N.OUT + '/paper_shap_compare.csv', index=False, encoding='utf-8-sig')
