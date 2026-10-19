# -*- coding: utf-8 -*-
"""Slope / Mean / RMS 세 묶음 켜기·끄기 (2^3 = 8 조합) — 교차검증 + 시험(LGB .7 + LR .3) + SHAP summary(beeswarm) 그림
사용: python noleak_toggle.py S1M1R1 S0M0R0 ...   (S=Slope, M=Mean, R=RMS, 1=켜기 0=끄기)
기본(항상 켜짐) = 순간값 제외 + 변화 피처(F2)에서 기울기·평균·RMS 성격 피처를 뺀 나머지 → 세 묶음과 겹치지 않음.
환자 단위 점이 그려지는 그림과 SHAP 값은 PC 폴더(noleak_out/toggle/)에만 저장."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from matplotlib import font_manager as fm
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
fam = pd.read_parquet('_work/noleak/fam_feats.parquet'); df = df.merge(fam, on='stay_id', how='left'); assert len(df) == 11000
sp = {k: df[df.split == k] for k in ('train', 'val', 'test')}; tr, te = sp['train'], sp['test']; yte = te.label.values
FAM = {k: [c for c in fam.columns if c.startswith('fam_%s_' % k)] for k in ('slope', 'mean', 'rms')}
DROP = set(G['vit_slope']) | {c for c in F2 if c.startswith(('nf4_uo_bin', 'nf4_uo_slope', 'nf4_uo_sqrt_slope', 'nf4_cr_slope'))} | {'tr_cr_slope_day', 'nf4_uo_sqrt_rmssd', 'hr_mean', 'sbp_mean', 'dbp_mean', 'mbp_mean', 'rr_mean', 'temp_c_mean'}
BASE = [c for c in F2 if c not in DROP]
for f in fm.findSystemFonts():
    if 'NotoSansCJK' in f: fm.fontManager.addfont(f)
kf = [n for n in {f.name for f in fm.fontManager.ttflist} if 'Noto Sans CJK' in n]
if kf: plt.rcParams['font.family'] = sorted(kf)[0]
plt.rcParams['axes.unicode_minus'] = False
OUTD = N.OUT + '/toggle'; os.makedirs(OUTD, exist_ok=True)
path = OUTD + '/toggle_results.json'; res = json.load(open(path)) if os.path.exists(path) else {}
y, grp = df.label.values, df.subject_id.values
T0 = time.time()
for code in sys.argv[1:]:
    on = {'slope': code[1] == '1', 'mean': code[3] == '1', 'rms': code[5] == '1'}
    feats = BASE + sum((FAM[k] for k in ('slope', 'mean', 'rms') if on[k]), [])
    t0 = time.time(); cvr = cv(feats)
    pl, ms = lg(feats); pe = .7 * rk(pl) + .3 * rk(lrp(feats, tr, te))
    rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
    bv = [roc_auc_score(yte[i], pe[i]) for i in boots if yte[i].min() != yte[i].max()]
    contrib = np.mean([m.booster_.predict(te[feats], pred_contrib=True) for m in ms], 0)
    vals, base = contrib[:, :-1], contrib[:, -1]
    share = pd.Series(np.abs(vals).mean(0), index=feats); share = 100 * share / share.sum()
    famshare = {k: round(float(share[FAM[k]].sum()), 1) if on[k] else 0.0 for k in FAM}; famshare['기본'] = round(float(share[BASE].sum()), 1)
    res[code] = {'Slope': on['slope'], 'Mean': on['mean'], 'RMS': on['rms'], 'n_features': len(feats), 'cv_auroc_lgb': round(cvr, 4),
                 'test_auroc_ens': round(roc_auc_score(yte, pe), 3), 'test_ci': [round(float(np.percentile(bv, 2.5)), 3), round(float(np.percentile(bv, 97.5)), 3)],
                 'test_auprc_ens': round(average_precision_score(yte, pe), 3), 'test_auroc_lgb': round(roc_auc_score(yte, pl), 3), 'shap_share_by_family': famshare,
                 'top10': [[k, round(float(v), 2)] for k, v in share.sort_values(ascending=False).head(10).items()]}
    json.dump(res, open(path, 'w'), ensure_ascii=False, indent=1)
    share.sort_values(ascending=False).round(3).to_csv(OUTD + '/shap_share_%s.csv' % code, header=['기여도(%)'], encoding='utf-8-sig')
    data = te[feats].copy()
    for c in cats:
        if c in data: data[c] = np.nan            # 범주형은 색 없이 회색
    ex = shap.Explanation(values=vals, base_values=base, data=data.astype(float).values, feature_names=feats)
    plt.figure(figsize=(10.5, 10.8)); shap.plots.beeswarm(ex, max_display=20, show=False, color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_xlabel('SHAP value  (← AKI 예측 점수 낮춤 | 높임 →, 로그오즈)', fontsize=10)
    lab = ' · '.join('%s %s' % (n, 'ON' if on[k] else 'OFF') for n, k in (('Slope', 'slope'), ('Mean', 'mean'), ('RMS', 'rms')))
    plt.title('AKI 위험모델 SHAP summary plot — %s\n(LightGBM 5회 평균, 시험 %d건, 특징 %d개, 시험 AUROC %.3f)' % (lab, len(te), len(feats), res[code]['test_auroc_ens']), fontsize=11.5)
    plt.gcf().text(0.01, 0.005, '빨강 = 값 높음, 파랑 = 값 낮음, 회색 = 결측·범주형. fam_* = 켜고 끄는 묶음(창 48h/24h/12h). 크레아티닌은 첫 측정 대비 비. 인과 아님.', fontsize=8, color='#555')
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUTD + '/shap_beeswarm_%s.png' % code, dpi=150); plt.close()
    N.log('%s | 특징 %3d | CV %.4f | 시험 앙상블 %.3f %s | 묶음 기여 %s | %.0f초' % (lab, len(feats), cvr, res[code]['test_auroc_ens'], res[code]['test_ci'], famshare, time.time() - t0))
    if time.time() - T0 > 120: print('시간 제한: 여기까지'); break
print('기본 특징', len(BASE), '개, 묶음', {k: len(v) for k, v in FAM.items()})
