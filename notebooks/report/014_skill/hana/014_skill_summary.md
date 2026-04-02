# 014 Skill Report (HANA)

## Setup
- Targets: 48
- Seeds: 5
- Density thresholds: dense >= 0.700, ultra <= 0.005
- Bucket method: full-dataset target nonzero rate using the same Stage 2 split helper

## Aggregate Validation
- Recomputed mean-of-mean MSE: 5.309734
- Saved mean-of-mean MSE: 5.309734
- MSE delta: +0.000000e+00
- Recomputed mean-of-mean skill: 0.151636
- Saved mean-of-mean skill: 0.151636
- Skill delta: +1.000604e-09

## Per-Drug Skill
- Valid-skill drugs: 42/48
- Positive-skill drugs: 20/42 (47.6%)
- Mean per-drug skill: 0.0880
- Median per-drug skill: 0.0000

## Dense Bucket Focus
- Dense drugs beating naive: 4/4

## Skill by Bucket
- dense: row_skill=0.1424, mean_drug_skill=0.1305, positive_targets=4/4
- sparse: row_skill=0.1524, mean_drug_skill=0.0585, positive_targets=14/30
- ultra: row_skill=0.4906, mean_drug_skill=0.1771, positive_targets=2/8

## Top Positive Drugs
- H02AB04: skill=0.8571, bucket=ultra, nonzero_rate=0.001, mean_y=0.00
- J01DD04: skill=0.5600, bucket=ultra, nonzero_rate=0.004, mean_y=0.00
- N01AX07: skill=0.3228, bucket=sparse, nonzero_rate=0.020, mean_y=0.02
- C01EA01: skill=0.2458, bucket=sparse, nonzero_rate=0.016, mean_y=0.03
- D11AH05: skill=0.2357, bucket=sparse, nonzero_rate=0.102, mean_y=0.15
- B05BB02: skill=0.2344, bucket=sparse, nonzero_rate=0.659, mean_y=4.83
- A11EA: skill=0.2242, bucket=sparse, nonzero_rate=0.091, mean_y=0.14
- R06AB04: skill=0.1803, bucket=sparse, nonzero_rate=0.474, mean_y=1.54
- H02AB02: skill=0.1607, bucket=dense, nonzero_rate=0.799, mean_y=10.86
- J01DC05: skill=0.1471, bucket=sparse, nonzero_rate=0.434, mean_y=3.15

## Worst Drugs
- H02AB08: skill=-0.1483, bucket=sparse, nonzero_rate=0.203, mean_y=0.28
- J01CR02: skill=-0.0442, bucket=sparse, nonzero_rate=0.155, mean_y=0.20
- B05XA03: skill=-0.0392, bucket=sparse, nonzero_rate=0.164, mean_y=0.22
- A03FA01: skill=-0.0236, bucket=sparse, nonzero_rate=0.233, mean_y=0.30
- A03AB02: skill=-0.0090, bucket=sparse, nonzero_rate=0.036, mean_y=0.06
- A04AA01: skill=-0.0074, bucket=sparse, nonzero_rate=0.010, mean_y=0.03
- N01AX10: skill=-0.0069, bucket=sparse, nonzero_rate=0.052, mean_y=0.07
- R05CB06: skill=-0.0062, bucket=sparse, nonzero_rate=0.053, mean_y=0.07
- N05CD08: skill=-0.0058, bucket=sparse, nonzero_rate=0.053, mean_y=0.07
- A02BA01: skill=-0.0031, bucket=sparse, nonzero_rate=0.059, mean_y=0.07