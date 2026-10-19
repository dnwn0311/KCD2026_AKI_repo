# -*- coding: utf-8 -*-
"""Slope·Mean·RMS 켜기 조합의 SHAP beeswarm 을 '한국어 설명 라벨'로 다시 그린다 (모델·SHAP 계산은 noleak_toggle.py 와 같음, 교차검증은 생략).
사용: python noleak_toggle_kor.py S1M1R1 [S0M0R0 ...]
출력: noleak_out/toggle/shap_beeswarm_한글_<코드>.png, 변수설명표_한글_<코드>.csv (변수명·한국어 이름·설명·단위·기여도%)  — 집계값만, 환자 행 없음
점이 그려지는 그림은 PC 폴더에만 저장."""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from matplotlib import font_manager as fm
src = open('noleak_toggle.py', encoding='utf-8').read()
exec(src.split("OUTD = N.OUT")[0])            # 데이터·묶음·BASE·FAM·글꼴 준비 (noleak_toggle.py 앞부분 그대로)
exec(open('noleak_kr_labels.py', encoding='utf-8').read())
OUTD = N.OUT + '/toggle'; res = json.load(open(OUTD + '/toggle_results.json'))
for code in sys.argv[1:]:
    on = {'slope': code[1] == '1', 'mean': code[3] == '1', 'rms': code[5] == '1'}
    feats = BASE + sum((FAM[k] for k in ('slope', 'mean', 'rms') if on[k]), [])
    pl, ms = lg(feats)
    contrib = np.mean([m.booster_.predict(te[feats], pred_contrib=True) for m in ms], 0); vals, base = contrib[:, :-1], contrib[:, -1]
    share = pd.Series(np.abs(vals).mean(0), index=feats); share = 100 * share / share.sum()
    info = {f: kr(f) for f in feats}
    tab = pd.DataFrame({'변수명(영어)': feats, '한국어 이름': [info[f]['short'] for f in feats], '데이터 설명': [info[f]['desc'] for f in feats], '단위': [info[f]['unit'] for f in feats],
                        '기여도(%)': share.values.round(3)}).sort_values('기여도(%)', ascending=False); tab.insert(0, '순위', range(1, len(tab) + 1))
    tab.to_csv(OUTD + '/변수설명표_한글_%s.csv' % code, index=False, encoding='utf-8-sig')
    data = te[feats].copy()
    for c in cats:
        if c in data: data[c] = np.nan
    ex = shap.Explanation(values=vals, base_values=base, data=data.astype(float).values, feature_names=[info[f]['short'] for f in feats])
    plt.figure(figsize=(11.5, 10.8)); shap.plots.beeswarm(ex, max_display=20, show=False, color_bar=True, plot_size=None)
    ax = plt.gca(); ax.set_xlabel('SHAP value  (← AKI 예측 점수 낮춤 | 높임 →, 로그오즈)', fontsize=10)
    ax.set_yticklabels([t.get_text().replace('Sum of', '그 외').replace('other features', '개 변수 합') for t in ax.get_yticklabels()], fontsize=10.5)
    lab = ' · '.join('%s %s' % (n, 'ON' if on[k] else 'OFF') for n, k in (('Slope', 'slope'), ('Mean', 'mean'), ('RMS', 'rms')))
    plt.title('AKI 위험모델 SHAP summary plot — %s\n(LightGBM 5회 평균, 시험 %d건, 특징 %d개, 시험 AUROC %.3f)' % (lab, len(te), len(feats), res[code]['test_auroc_ens']), fontsize=11.5)
    plt.gcf().text(0.01, 0.005, '빨강 = 값 높음, 파랑 = 값 낮음, 회색 = 결측·범주형(병동). 기준 시점 = ICU 입실 후 48시간, 기울기 = 하루당, 소변량 = mL/kg/h. 인과 아님.', fontsize=8, color='#555')
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(OUTD + '/shap_beeswarm_한글_%s.png' % code, dpi=150); plt.close()
    print(code, '완료', len(feats), '개; 미매핑', [f for f in feats if info[f]['short'] == f])
    print(tab.head(25)[['순위', '변수명(영어)', '한국어 이름', '기여도(%)']].to_string(index=False))
