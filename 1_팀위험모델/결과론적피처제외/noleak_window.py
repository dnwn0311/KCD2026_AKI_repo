# -*- coding: utf-8 -*-
"""1단계: 코호트 환자의 입실~기준시점(48h) 창 안 활력징후·검사 원자료를 추려 _work/noleak/ 에 저장 (환자 행은 출력하지 않음)
사용: python noleak_window.py vitals|labs
"""
import sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
kind = sys.argv[1]
t0 = time.time()
c = pd.read_parquet('_work/team_fixed/cohort_final.parquet', columns=['stay_id', 'hadm_id', 'intime', 'time_zero'])
c = c.drop_duplicates('hadm_id').set_index('hadm_id')
hs = set(c.index)
pf = pq.ParquetFile('_work/%s.parquet' % kind)
out = []
for g in range(pf.num_row_groups):
    d = pf.read_row_group(g, columns=['hadm_id', 'code', 'value', 't']).to_pandas()
    d = d[d.hadm_id.isin(hs)]
    if len(d):
        m = c.loc[d.hadm_id.values]
        h = (d.t.values - m.intime.values) / np.timedelta64(1, 'h')
        ok = (h >= 0) & (d.t.values < m.time_zero.values)
        out.append(pd.DataFrame({'stay_id': m.stay_id.values[ok], 'code': d.code.values[ok].astype('int8'),
                                 'hour': h[ok].astype('float32'), 'value': d.value.values[ok].astype('float32')}))
    if time.time() - t0 > 150:
        print('시간 초과 직전 중단 g=%d/%d' % (g, pf.num_row_groups)); break
r = pd.concat(out, ignore_index=True)
r.to_parquet('_work/noleak/win_%s.parquet' % kind, index=False)
print(kind, '행', len(r), '체류', r.stay_id.nunique(), '코드', r.code.nunique(), '소요 %.0f초' % (time.time() - t0))
