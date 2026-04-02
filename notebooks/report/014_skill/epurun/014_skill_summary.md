# 014 Skill Report (Epurun)

## Setup
- Targets: 248
- Seeds: 5
- Density thresholds: dense >= 0.700, ultra <= 0.005
- Bucket method: full-dataset target nonzero rate using the same Stage 2 split helper

## Aggregate Validation
- Recomputed mean-of-mean MSE: 0.919546
- Saved mean-of-mean MSE: 0.919546
- MSE delta: -2.220446e-16
- Recomputed mean-of-mean skill: -0.109212
- Saved mean-of-mean skill: -0.109212
- Skill delta: -6.369884e-10

## Per-Drug Skill
- Valid-skill drugs: 223/248
- Positive-skill drugs: 18/223 (8.1%)
- Mean per-drug skill: -0.8137
- Median per-drug skill: -0.0703

## Dense Bucket Focus
- Dense drugs beating naive: 5/97

## Skill by Bucket
- dense: row_skill=-0.0670, mean_drug_skill=-0.1877, positive_targets=5/97
- sparse: row_skill=-0.2212, mean_drug_skill=-1.3272, positive_targets=13/123
- ultra: row_skill=0.0000, mean_drug_skill=0.0000, positive_targets=0/3

## Top Positive Drugs
- R03CC08: skill=0.4706, bucket=sparse, nonzero_rate=0.006, mean_y=0.01
- A06AB02: skill=0.4000, bucket=sparse, nonzero_rate=0.014, mean_y=0.01
- N06BA04: skill=0.3200, bucket=sparse, nonzero_rate=0.012, mean_y=0.01
- N04AA04: skill=0.2105, bucket=sparse, nonzero_rate=0.010, mean_y=0.01
- B05BA01: skill=0.1633, bucket=sparse, nonzero_rate=0.036, mean_y=0.04
- A11GA01: skill=0.0458, bucket=sparse, nonzero_rate=0.143, mean_y=0.27
- A02AD03: skill=0.0282, bucket=dense, nonzero_rate=0.761, mean_y=2.00
- B05BB02: skill=0.0146, bucket=dense, nonzero_rate=1.000, mean_y=9.16
- J01FA09: skill=0.0139, bucket=sparse, nonzero_rate=0.041, mean_y=0.04
- J01XA02: skill=0.0108, bucket=sparse, nonzero_rate=0.117, mean_y=0.17

## Worst Drugs
- V03AB: skill=-31.3880, bucket=sparse, nonzero_rate=0.080, mean_y=0.08
- N04BC05: skill=-23.9267, bucket=sparse, nonzero_rate=0.130, mean_y=0.20
- C05CX02: skill=-14.0000, bucket=sparse, nonzero_rate=0.053, mean_y=0.05
- P01BA02: skill=-14.0000, bucket=sparse, nonzero_rate=0.043, mean_y=0.04
- H01BA02: skill=-13.7844, bucket=sparse, nonzero_rate=0.159, mean_y=0.16
- H03BB02: skill=-13.3490, bucket=sparse, nonzero_rate=0.474, mean_y=0.47
- A10BK01: skill=-8.0928, bucket=sparse, nonzero_rate=0.104, mean_y=0.10
- A10BK03: skill=-6.0405, bucket=sparse, nonzero_rate=0.383, mean_y=0.80
- N03AX09: skill=-5.4286, bucket=sparse, nonzero_rate=0.031, mean_y=0.03
- A11EB: skill=-3.9598, bucket=sparse, nonzero_rate=0.392, mean_y=1.57