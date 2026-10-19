# -*- coding: utf-8 -*-
"""F2 + RMS(소변·크레아티닌 14개) 앙상블 교차검증과 시험 평가 (F2 앙상블과 같은 방식)"""
import json, warnings
exec(open('noleak_flow_report.py').read().split("R = {}")[0])
rm = pd.read_parquet('_work/noleak/rms_feats.parquet'); df = df.merge(rm, on='stay_id', how='left'); assert len(df) == 11000
d = df.reset_index(drop=True); sp = {k: df[df.split == k] for k in ('train', 'val', 'test')}; tr, te = sp['train'], sp['test']; yte = te.label.values
RS = [c for c in rm.columns if c.split('_')[1] in ('uo', 'cr')]
FR = F2 + RS
out = {}
for nm, f in (('F2', F2), ('F2+RMS소변·크레아티닌', FR), ('F2+RMS전체', F2 + [c for c in rm.columns if c != 'stay_id'])):
    cvr = cv_ens(f)
    pl, _ = lg(f); pe = .7 * rk(pl) + .3 * rk(lrp(f, tr, te))
    rng = np.random.RandomState(0); boots = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
    v = [roc_auc_score(yte[i], pe[i]) for i in boots if yte[i].min() != yte[i].max()]
    out[nm] = {'n': len(f), 'cv': cvr, 'test_auroc': round(roc_auc_score(yte, pe), 3), 'ci': [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)], 'test_auprc': round(average_precision_score(yte, pe), 3)}
    print(nm, out[nm], flush=True); json.dump(out, open(N.OUT + '/rms_final.json', 'w'), ensure_ascii=False)
