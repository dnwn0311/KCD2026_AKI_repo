# -*- coding: utf-8 -*-
"""F3: 최근 구간 수준이 섞인 절대 소변 구간값(nf4_uo_bin*)까지 제거한 '변화 전용' 변형 — 교차검증 + 시험"""
import json, warnings
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
F3 = [c for c in F2 if not c.startswith('nf4_uo_bin')]
r = cv_ens(F3); print('F3 변화 전용 %d개 교차검증(2회):' % len(F3), r, flush=True)
pl, _ = lg(F3); pe = .7 * rk(pl) + .3 * rk(lrp(F3, tr, te))
rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
v = [roc_auc_score(yte[i], pe[i]) for i in boots if yte[i].min() != yte[i].max()]
print('F3 시험 AUROC %.3f [%.3f, %.3f] AUPRC %.3f' % (roc_auc_score(yte, pe), np.percentile(v, 2.5), np.percentile(v, 97.5), average_precision_score(yte, pe)))
json.dump({'cv': r, 'test_auroc': round(roc_auc_score(yte, pe), 3)}, open(N.OUT + '/flow_f3.json', 'w'))
