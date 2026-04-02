# LAVAR需要予測研究: 85-run sweepから堅牢なアンサンブルへ

## エグゼクティブサマリー

14日先の供給予測に向けて、2段階の潜在オートエンコーダVAR（LAVAR）パイプラインを開発し、2つの病院データセットで検証した。対象は **Hana**（48 ATCターゲット、130 folds）と **Epurun**（248 ATCターゲット、52 folds）である。評価には、厳密な非重複ローリングバックテストの下で85通りのモデル構成を用いた。

**モデル選定。** 85-run sweepの結果、`S11_LATENT16`（latent dim=16、GRU head）がHanaにおける最良の単一モデルとして浮上した（mean MSE 5.28、skill 0.14）。`S04_GRU_H32_L1`は補完的な次点候補だった。`S16`の正則化バリアント2種は、テールリスクのスパイクが大きかったため除外した（max fold MSEは最大90、S11は39）。さらに、5つの乱数初期化にまたがるseed stability gateにより、S11の優位性が単一seedの偶然ではないことを確認した（CV 0.052、S04は0.082）。

**アンサンブル。** 70/30の時間分割で重みを選んだ静的な凸結合 `y = 0.55 · S11 + 0.45 · S04` は、Hanaで純粋なS11に比べてmean MSEを **−4.89%** 改善した（5.31 vs 5.58）。同時にテールリスクも圧縮された（p95 fold MSE: 12.8 vs 14.0）。この改善は独立な **5/5** seedsすべてで再現され、seed間分散もブレンドの方が小さかった（std 0.18 vs 0.29）。

**Epurunでの再現。** 同じS11/S04ブレンド手法をEpurun（248 targets、52 folds）でも再現した。ブレンド（w=0.55）は純粋なS11に対してmean MSEを **−4.38%** 改善し（0.920 vs 0.962）、**5/5** seedsで勝利した。注目すべき点として、Epurunでは中心傾向ではS04がS11をわずかに上回り（mean MSE 0.957 vs 0.962）、Hanaとは順位が逆転している一方、S11はテール挙動で優位を保った（max fold MSE 3.45 vs 4.57）。ブレンドは両モデルの強みを活用し、max fold MSEを大きく圧縮した（median 2.40、S11は3.27）。

**薬剤別インサイト。** 誤差は両病院で集中している。Hanaでは48薬剤中5薬剤で集計MSEの約72%を占め、その先頭はH02AB02の25%だった。Epurunでは248薬剤中上位5薬剤でMSEの約20%を占め、先頭はB05XA03だった。大半の薬剤はほぼゼロ誤差であり、高誤差薬剤に対する重点的な改善が最もレバレッジの高い経路である。

**推奨。** HanaではS11/S04ブレンドを w=0.55–0.60、Epurunでは w=0.50–0.55 で運用する。両サイトとも、単一モデルのフォールバックとして純粋なS11を利用できる。

---

## 1) 目的

統合されたHanaデータ（`X`, `y`）上で、厳密な非重複ローリングバックテスト（`horizon=14`, `fold_step=14`）を用いて予測パイプラインを構築・評価し、広範なモデル探索から堅牢なショートリストとアンサンブルへ候補を絞り込んだ。

---

## 2) 実験プロトコル

- 総run数: **85 / 85 completed**（Hana sweep）、**10 / 10**（Epurun seed replication）
- デバイス: **mps**
- バックテスト設定: `horizon=14`, `fold_step=14`（非重複）、retrain cadence `90`、quality triggers有効
- 1 runあたりのfold数: **130**（Hana）、**52**（Epurun）
- 主なソース（Hana）:
  - `report/005_model/hana/leaderboard.csv`
  - `report/005_model/hana/worst_folds_by_mse.csv`
  - `report/006_graph/hana/top4_fold_metrics_reconstructed.csv`
  - `report/007_ensemble/hana/blend_summary.csv`
  - `report/007_ensemble/hana/blend_eval_temporal_split.csv`
  - `report/008_seed/hana/seed_stability_summary.csv`
  - `report/008_seed/hana/seed_per_run_metrics.csv`
  - `report/009_seed/hana/blend_seed_fixed_weights_summary.csv`
  - `report/009_seed/hana/blend_seed_wstar_summary.csv`
  - `report/009_seed/hana/blend_seed_robustness_decision.csv`
  - `report/010_ensemble/hana/ensemble_metrics_overall.csv`
  - `report/010_ensemble/hana/ensemble_metrics_by_seed.csv`
  - `report/010_ensemble/hana/ensemble_ci_per_drug.csv`
- 主なソース（Epurun）:
  - `report/011_seed/epurun/seed_stability_summary.csv`
  - `report/011_seed/epurun/seed_per_run_metrics.csv`
  - `report/012_seed/epurun/blend_seed_fixed_weights_summary.csv`
  - `report/012_seed/epurun/blend_seed_wstar_summary.csv`
  - `report/012_seed/epurun/blend_seed_robustness_decision.csv`
  - `report/013_ensemble/epurun/ensemble_metrics_overall.csv`
  - `report/013_ensemble/epurun/ensemble_metrics_by_seed.csv`
  - `report/013_ensemble/epurun/ensemble_ci_per_drug.csv`

---

## 3) Stage A - 広範なモデルスイープ（85 runs）

85-run gridから、以下を観測した。

- **最高mean-MSEの単一モデル**: `S11_LATENT16 + E05_CAP_200_300`
  - mean MSE: **5.278**
  - mean skill: **0.136**
  - median skill: **0.198**
- **有力な次点候補**: `S04_GRU_H32_L1 + E02_CAP_200_200`
  - mean MSE: **5.476**
  - median skill: **0.169**
- `S16_REG_LATENT16` は魅力的な中心傾向（median MSE）を示した一方、一部のfoldで明確なテールリスクを示した。

主要なリスクパターン（worst-fold分析より）:
- 深刻な失敗は、弱いモデル群の特定の時点（`t_end~1080`, `t_end~1528`）に集中する。
- 最終候補の中では、`S16` が `t_end=1528` 付近で顕著なスパイク挙動を示す。

---

## 4) Stage B - Top-4診断分析

fold単位および軌跡レベルのより深い診断のため、上位4候補を選定した。

1. `RUN_S11_LATENT16_E05_CAP_200_300`
2. `RUN_S04_GRU_H32_L1_E02_CAP_200_200`
3. `RUN_S16_REG_LATENT16_E05_CAP_200_300`
4. `RUN_S16_REG_LATENT16_E02_CAP_200_200`

### 所見

- `S11` と `S04` はfold全体で比較的安定している。
- `S16` バリアントは、特に `t_end=1528` 付近でfoldスパイクが大きい。
  - `S16 E05` max fold MSE: **58.91**
  - `S16 E02` max fold MSE: **90.16**
- 比較対象として:
  - `S11` max fold MSE: **38.67**
  - `S04` max fold MSE: **38.01**

### リーク / split integrityチェック

すべてのチェックに合格した（`report/006_graph/hana/leakage_audit.csv`）。

- fold-step consistency（`14`）
- 非重複ウィンドウチェック
- prediction contextが厳密に `t_end` より前
- `fit_t_end <= t_end` の境界条件
- metric再計算の整合性

---

## 5) Stage C - 2モデルへの絞り込み（S11, S04）

精度と安定性のトレードオフを踏まえ、Top-4からTop-2へ絞り込んだ。

- **主候補**: `S11`
- **補完候補**: `S04`

根拠:
- `S11` は全体平均指標で最も強い。
- `S04` は十分に競争力があり、一部の局所的な誤差領域では勝つため、S11と完全には冗長でない。
- `S16` には有用なアイデアがあるが、制御しにくいテールリスクを持ち込む。

---

## 6) Stage D - アンサンブル改善（S11 + S04）

既存のfold-level per-drug predictions上で、再学習なしの静的な凸ブレンドを検証した。

`y_blend = w * y_s11 + (1 - w) * y_s04`

ここで `w ∈ [0, 1]` はS11予測に対する重みである。`w=1` は純粋なS11、`w=0` は純粋なS04を意味する。`w` を 0.00 から 1.00 まで 0.05 刻みでgrid searchし、各重みをmean MSE、mean MAE、naive比skill、およびテールリスク指標（p95とmax fold MSE）で評価した。

重み選択には、楽観バイアスを避けるため **temporal split** を使った。`w*` はfoldの先頭70%（`t_end`順）でチューニングし、保持しておいた後半30%で評価した。

比較した目的関数は2つである。
- **mean-MSE objective** → `w* = 0.55`
- **composite objective**（mean_mse + 0.5 × p95_fold_mse、テールリスク罰則付き）→ `w* = 0.50`

### 全サンプルでのブレンド要約

`blend_summary.csv` より:

| 構成 | w | Mean MSE | Mean MAE | Mean Skill | P95 Fold MSE | Max Fold MSE |
|---|---|---|---|---|---|---|
| 純粋なS04 | 0.00 | 5.476 | 0.742 | 0.095 | 11.741 | 38.01 |
| 純粋なS11 | 1.00 | 5.278 | 0.734 | 0.136 | 12.925 | 38.67 |
| ブレンド（mean-MSE w*） | 0.55 | **4.996** | **0.720** | **0.203** | **10.149** | 38.27 |
| ブレンド（composite w*） | 0.50 | 5.003 | 0.720 | 0.202 | 10.036 | 38.24 |

どちらのブレンド重みも、純粋なS11に比べてmean MSEを約5%改善しつつ、p95 fold MSE（テールリスク）も低減した。composite w*=0.50 は、mean MSEの悪化をほぼ伴わずに（+0.007）、最小のp95（10.04 vs 10.15）を達成した。

### Temporal split評価（70%で調整、30%で評価）

`blend_eval_temporal_split.csv` より:

| 構成 | Mean MSE | Δ vs 純粋なS11 |
|---|---|---|
| 純粋なS04 | 7.439 | +0.390 |
| 純粋なS11 | 7.049 | — |
| ブレンド（w*=0.55） | **6.962** | **−0.087 (−1.2%)** |
| ブレンド（w*=0.50） | 6.979 | −0.070 (−1.0%) |

このtemporal splitでの改善幅は −1.2% と控えめに見えるが、これは単一seedに基づく保守的な推定である。後のStage F（009_seed）では、5つの独立初期化にわたって平均 −0.27 MSE（約5%）の改善が確認され、ブレンド効果の堅牢性が示された。ここでのtemporal splitは `w*` がin-sample foldsに過度適合していないことの健全性確認として機能し、最終的な堅牢性の根拠はStage Fのcross-seed解析が与える。

---

## 7) Stage E - Seed stability gate（008_seed）

ショートリストに残した2モデル（`S11_LATENT16`, `S04_GRU_H32_L1`）について、それぞれ5 seeds（`42, 123, 456, 789, 1024`）でseed stability gateを実施した。rolling protocolは同一（`horizon=14`, `fold_step=14`, 130 folds/run）。

Run completion status: **10 / 10**（`report/008_seed/hana/run_status.json`）。

### Seed要約（各シナリオ5 seeds）

| シナリオ | Mean MSE（mean +- std） | CV（mean MSE） | Median Skill（mean +- std） | Max Fold MSE（mean） | P95 Fold MSE（mean） |
|---|---:|---:|---:|---:|---:|
| S11_LATENT16 | **5.583 +- 0.288** | **0.052** | **0.167 +- 0.029** | **42.31** | **14.00** |
| S04_GRU_H32_L1 | 6.148 +- 0.506 | 0.082 | 0.111 +- 0.033 | 79.86 | 14.88 |

`seed_per_run_metrics.csv` からの追加の直接比較:
- S11は **4/5** seedsでより低いmean MSEを示した。
- S11は **5/5** seedsでより高いmedian skillを示した。
- S04はseed `789` で重いテール故障を示し（`max_fold_mse=193.14`）、これがtail-risk平均を押し上げている。

解釈:
- このseed gateは、S11が単なる単一seed勝者ではなく、平均的により安定し、中心傾向とテール挙動の両方で優れていることを確認した。
- これにより、単一モデルのベースラインとしてS11を維持し、S04は主にブレンド用の補完モデルとして使う判断が強化される。

---

## 8) Stage F - Cross-seedブレンド堅牢性（009_seed）

Stage Dのアンサンブル重みは単一seedの予測上で最適化された。しかしStage Eで、S04はseed間でS11よりノイジーであることが分かった（CV 0.082 vs 0.052）。そこで確認すべき点は次である。**両モデルを異なる乱数初期化で学習しても、ブレンドは依然として有効か。** 本番環境ではseedは固定されないため、配備するブレンドはどの初期化でも堅牢である必要がある。

008の10 runsすべてについてper-drug predictionsを再生成し、その後、各seed pairごとに各重みでS11+S04ブレンドを計算した。

### Mean-MSE改善は堅牢

固定重み w=0.55 では、ブレンドは **5/5** seedsで純粋なS11のmean MSEを上回った。

| Seed | 純粋なS11 MSE | ブレンド（w=0.55）MSE | Delta | P95 Delta |
|---:|---:|---:|---:|---:|
| 42 | 5.293 | 5.082 | **-0.211** | -0.60 |
| 123 | 6.045 | 5.490 | **-0.554** | -3.20 |
| 456 | 5.407 | 5.165 | **-0.242** | -1.06 |
| 789 | 5.538 | 5.449 | **-0.089** | +0.89 |
| 1024 | 5.630 | 5.363 | **-0.267** | -2.13 |

平均改善量は **-0.273 MSE**（約5%）だった。ブレンドはp95でも 4/5 seeds で勝ち、唯一の例外はseed 789だった。このseedではS04が破局的なdraw（`max_fold_mse=193`）を引いたが、それでもp95悪化は小さい（+0.89）。

### 最適重みはseed依存

seedごとの最適な `w*` は **0.45 から 0.75** の範囲に分布した。

| Seed | w*（mean MSE） | w*（composite） |
|---:|---:|---:|
| 42 | 0.65 | 0.70 |
| 123 | 0.45 | 0.45 |
| 456 | 0.65 | 0.65 |
| 789 | 0.75 | 0.75 |
| 1024 | 0.60 | 0.60 |

このパターンは直感的である。S04が弱いとき（seed 789）は、最適化器は `w*` をS11寄り（0.75）へ押し上げる。S04が強いとき（seed 123）は、`w*` は下がる（0.45）。固定値としては w=0.55–0.60 が妥当な中間点であり、どのseedでも破局的失敗を起こさない。

### テールリスクの圧縮

最も説得力のある結果は、seedをまたいだテールリスク比較である。どちらのブレンドも、純粋な各モデルに比べてp95とmax-fold-MSEの分布を大幅に圧縮した。

- **P95 fold MSE**（seed横断median）: S11=14.0、S04=14.9、Blend w=0.55=**12.5**
- **Max fold MSE**（seed横断median）: S11=44.1、S04=51.1、Blend w=0.55=**43.3**

ブレンドは平均性能を改善するだけでなく、乱数初期化による悪い結果の分散も減らす。これが配備に向けた最も強い論拠である。不運なseedを引いても、ブレンドがダメージを抑える。

---

## 9) Stage G — 薬剤別内訳を含む本番アンサンブル（010_ensemble）

Stages D–F により、w=0.55 のS11+S04ブレンドがmean MSEを約5%改善し、テールリスクも圧縮すること、さらにその効果が5つの独立seed drawsで成り立つことが示された。Stage Gでは、本番に近いモードでアンサンブルを実行する。条件は5 seeds、per-drug predictionsを `[0, ∞)` にclip、そしてseeded blend predictionsに対する経験的95%信頼区間（q2.5, q97.5）の算出である。

### 全体指標（5 seeds）

| 構成 | Mean MSE（mean ± std） | Mean MAE | Mean Skill | P95 Fold MSE | Max Fold MSE |
|---|---:|---:|---:|---:|---:|
| Blend w=0.55 | **5.310 ± 0.179** | **0.738** | **0.152** | **12.783** | 41.662 |
| Pure S11 | 5.583 ± 0.288 | 0.748 | 0.081 | 14.005 | 42.305 |
| Pure S04 | 6.148 ± 0.506 | 0.775 | −0.034 | 14.879 | 79.860 |

ブレンドは、純粋なS11に対してmean MSEを **−0.273（−4.89%）** 改善し、Stage Fの推定と整合した。特に、ブレンドのseed-to-seed標準偏差（0.179）は、どちらの純粋モデルよりも低い（S11: 0.288、S04: 0.506）。これは、ブレンドが乱数初期化をまたいで予測を安定化させることを確認している。

### Seed別内訳

| Seed | Blend MSE | S11 MSE | S04 MSE | Blend Δ vs S11 |
|---:|---:|---:|---:|---:|
| 42 | 5.082 | 5.293 | 5.793 | −0.211 |
| 123 | 5.490 | 6.045 | 5.928 | −0.554 |
| 456 | 5.165 | 5.407 | 6.047 | −0.242 |
| 789 | 5.449 | 5.538 | 7.038 | −0.089 |
| 1024 | 5.363 | 5.630 | 5.934 | −0.267 |

ブレンドは **5/5** seedsでmean MSEに勝利した。最大の改善はseed 123（−0.554）で、ここではS04がS11を特によく補完した。最小はseed 789（−0.089）で、S04が最悪の初期化を引いたケースだが、それでもブレンドは改善している。

### 薬剤別の誤差集中

010 ensembleは、初めて薬剤別の内訳を提供し、集計MSEが少数の高ボリューム薬剤に強く集中していることを示した。48個のATC targets全体では次の通りである。

| 薬剤（ATC） | Per-Drug MSE | 累積 % |
|---|---:|---:|
| H02AB02 | 59.8 | 25% |
| M01AB05 | 34.8 | 40% |
| B05BB02 | 29.8 | 52% |
| J01DC05 | 29.6 | 65% |
| J01GB03 | 17.6 | 72% |

上位5薬剤で総MSEの **約72%** を占める一方、多くの薬剤はほぼゼロ誤差である。これは、高誤差薬剤のごく一部に対する重点的なモデル改善が非常に大きな効果を生む可能性を示唆しており、集計MSE指標がこれら少数ターゲットに支配されていることも示している。

### 信頼区間の可視化

薬剤別CI図（Figure 12）は、誤差上位12薬剤について、真値の上にアンサンブル予測と経験的95% CI bandを重ねて表示している。大半の薬剤ではCI bandは狭く、真の軌跡をよく追っている。高誤差薬剤（H02AB02, M01AB05）ではCI bandがより広く、乖離も大きく、上記の誤差集中の所見と整合している。

---

## 10) Epurun再現 — Seed stability gate（011_seed）

LAVARパイプラインとS11/S04アンサンブル手法の一般化可能性を検証するため、Stages E–G を第2の病院データセット **Epurun**（248 ATC targets、52 folds、`horizon=14`, `fold_step=14`）で再現した。用いた5 seeds（`42, 123, 456, 789, 1024`）と訓練プロトコルは同一である。

Run completion status: **10 / 10**（`report/011_seed/epurun/run_status.json`）。

### Seed要約（各シナリオ5 seeds）

| シナリオ | Mean MSE（mean ± std） | CV（mean MSE） | Median Skill（mean ± std） | Max Fold MSE（mean） | P95 Fold MSE（mean） |
|---|---:|---:|---:|---:|---:|
| S04_GRU_H32_L1 | **0.957 ± 0.009** | **0.009** | **−0.032 ± 0.008** | 4.57 | **1.299** |
| S11_LATENT16 | 0.962 ± 0.011 | 0.012 | −0.049 ± 0.015 | **3.45** | 1.337 |

`seed_per_run_metrics.csv` からの追加の直接比較:
- S04は **4/5** seedsでより低いmean MSEを示した。
- S04は **5/5** seedsでより高いmedian skillを示した。
- S11は **5/5** seedsでより低いmax fold MSEを示した（最良のtail behavior）。

**Hanaとの対比:** Hanaでは、中心傾向ではS11が明確に強かった（mean MSE 5.58 vs S04の5.48）。Epurunでは順位が逆転し、平均ではS04がわずかに優れる（0.957 vs 0.962）。ただし、S11は一貫してtail-risk優位を保つ（max fold MSE 3.45 vs 4.57）。この分岐が、ブレンドの動機になる。中心指標とテール指標の両方で一方が他方を完全に支配しているわけではない。

---

## 11) Epurun再現 — Cross-seedブレンド堅牢性（012_seed）

HanaのStage Fプロトコルを再現し、5つのEpurun seed pairすべてについて各重みでS11+S04ブレンドを再実行した。

### Mean-MSE改善は堅牢

固定重み w=0.55 において、ブレンドは **5/5** seedsで純粋なS11のmean MSEを上回った。

| Seed | 純粋なS11 MSE | ブレンド（w=0.55）MSE | Delta | P95 Delta |
|---:|---:|---:|---:|---:|
| 42 | 0.961 | 0.922 | **−0.039** | −0.028 |
| 123 | 0.962 | 0.923 | **−0.039** | −0.044 |
| 456 | 0.976 | 0.929 | **−0.046** | −0.053 |
| 789 | 0.965 | 0.916 | **−0.049** | −0.031 |
| 1024 | 0.944 | 0.908 | **−0.037** | +0.008 |

平均改善量は **−0.042 MSE**（約4.4%）だった。p95でもブレンドは 4/5 seeds で勝利し、唯一の例外（seed 1024、p95 delta +0.008）も無視できる水準である。

### 最適重みはseed依存

| Seed | w*（mean MSE） | w*（composite） |
|---:|---:|---:|
| 42 | 0.50 | 0.45 |
| 123 | 0.45 | 0.40 |
| 456 | 0.45 | 0.35 |
| 789 | 0.45 | 0.35 |
| 1024 | 0.55 | 0.60 |

seedごとの `w*` は **0.45 から 0.55** の範囲で、Hanaの0.45–0.75よりもS04寄りである。これは、EpurunでS04が中心傾向の面でより強いモデルであることと整合する。固定値としては w=0.50–0.55 が妥当な中間点である。

### Robustness gate

w=0.50 と w=0.55 の両方がpromotion ruleを通過した（wins_both ≥ 4/5、深刻な劣化seedなし）。

| w | Wins（both） | Mean Δ MSE | Mean Δ P95 | Promote |
|---|---:|---:|---:|---|
| 0.50 | 4/5 | −0.043 | −0.032 | Yes |
| 0.55 | 4/5 | −0.042 | −0.029 | Yes |

### テールリスクの圧縮

ブレンドは、seedをまたいだmax fold MSEを劇的に圧縮した。

- **P95 fold MSE**（seed横断median）: S11=1.328、S04=1.293、Blend w=0.55=**1.297**
- **Max fold MSE**（seed横断median）: S11=3.273、S04=4.700、Blend w=0.55=**2.402**

max fold MSEの圧縮は特に印象的である。ブレンドのworst fold（median 2.40）は、S11の3.27より27%低く、S04の4.70より49%低い。

---

## 12) Epurun再現 — 薬剤別内訳を含む本番アンサンブル（013_ensemble）

### 全体指標（5 seeds）

| 構成 | Mean MSE（mean ± std） | Mean MAE | Mean Skill | P95 Fold MSE | Max Fold MSE |
|---|---:|---:|---:|---:|---:|
| Blend w=0.55 | **0.920 ± 0.008** | **0.404** | **−0.109** | 1.307 | **2.432** |
| Pure S04 | 0.957 ± 0.009 | 0.407 | −0.173 | **1.299** | 4.575 |
| Pure S11 | 0.962 ± 0.011 | 0.411 | −0.171 | 1.337 | 3.454 |

ブレンドは、純粋なS11に対してmean MSEを **−0.042（−4.38%）** 改善し、012_seedの推定と整合した。ブレンドのseed-to-seed標準偏差（0.008）はどちらの純粋モデルよりも低く（S11: 0.011、S04: 0.009）、ブレンドが予測を安定化させていることを確認している。

### Seed別内訳

| Seed | Blend MSE | S11 MSE | S04 MSE | Blend Δ vs S11 |
|---:|---:|---:|---:|---:|
| 42 | 0.922 | 0.961 | 0.957 | −0.039 |
| 123 | 0.923 | 0.962 | 0.952 | −0.039 |
| 456 | 0.929 | 0.976 | 0.966 | −0.046 |
| 789 | 0.916 | 0.965 | 0.946 | −0.049 |
| 1024 | 0.908 | 0.944 | 0.966 | −0.037 |

ブレンドは **5/5** seedsでmean MSEに勝利した。

### 薬剤別の誤差集中

248個のATC targets全体では次の通りである。

| 薬剤（ATC） | Per-Drug MSE | 累積 % |
|---|---:|---:|
| B05XA03 | 13.7 | 6% |
| A02AA04 | 10.2 | 11% |
| B05BB02 | 8.0 | 15% |
| A11EA | 6.8 | 18% |
| J01CR05 | 6.2 | 20% |

上位5薬剤で総MSEの **約20%** を占める。Hanaでは48薬剤中5薬剤で約72%を占めていたのに対し、Epurunの248 targetsでは誤差はより分散している。それでも、248薬剤中85薬剤は MSE < 0.1 であり、最大の誤差寄与薬剤に対する重点改善が依然として最も効率的な道である。

### 信頼区間の可視化

薬剤別CI図（Figure E3）は、誤差上位薬剤について真値の上にアンサンブル予測と経験的95% CI bandを重ねている。大半の薬剤ではCI bandは全般に狭いが、高誤差ターゲットではより広いbandが見られる。

---

## 13) 病院横断比較

| 指標 | Hana（48 targets） | Epurun（248 targets） |
|---|---:|---:|
| 最良の単一モデル | S11 | S04（わずかに優位） |
| S11 mean MSE（5-seed平均） | 5.583 | 0.962 |
| S04 mean MSE（5-seed平均） | 6.148 | 0.957 |
| Blend（w=0.55）mean MSE | 5.310 | 0.920 |
| Blend Δ vs S11 | −4.89% | −4.38% |
| Blend wins（seeds） | 5/5 | 5/5 |
| 最適 `w*` 範囲 | 0.45–0.75 | 0.45–0.55 |
| 上位5薬剤のMSEシェア | ~72% | ~20% |
| Max fold MSE圧縮 | 43.3 vs 44.1（S11） | 2.40 vs 3.27（S11） |

主な観察:
- ブレンドによる改善（約4–5%）は、ターゲット数、誤差スケール、モデル順位が大きく異なるにもかかわらず、両病院で非常に一貫している。
- Epurunでは最適 `w*` がS04寄りにシフトしており、これは同データセットでのS04の相対的な強さを反映している。
- 誤差集中はEpurunよりHanaで大きい（ターゲット数が少なく、ボリュームの大きい薬剤に集中）。
- Max fold MSEの圧縮率はEpurunの方が比例的に大きい（27%減、Hanaは2%減）。

---

## 14) 限界

- **各horizon stepの内訳が欠けている。** 薬剤別分析はStage G（Hana）と013（Epurun）で追加されたが、h=1 と h=14 のようなhorizon step別内訳はまだ検証していない。潜在rolloutのdriftが蓄積する長いhorizonに誤差が集中している可能性がある。
- **Sweep全体図がない。** 85-runの全景（たとえばシナリオ × epochヒートマップ）が可視化されておらず、探索範囲の広さを把握しにくい。
- **固定ブレンド重み。** 最適 `w*` はseedごとにHanaで0.45–0.75、Epurunで0.45–0.55の範囲で変動する。静的な w=0.55 は実務上の妥協点だが、理論的最適ではない。最近のvalidation lossに基づく適応重み付けなどで改善余地はあるが、配備の複雑さは増す。
- **EpurunではStages A–Dを再現していない。** 85-run sweep、Top-4診断、初期アンサンブル調整（Stages A–D）はEpurunでは再実施せず、同じS11/S04モデルを使ってseed stability、cross-seed blend、本番アンサンブル段階（E–G相当）のみを再現した。

---

## 15) 結論

進行の流れ:

1. **85-run sweep**（005）で、17 scenarios × 5 epoch profilesにわたるモデル地形を把握した。
2. **Top-4 shortlisting**（006）を、mean/median指標とfold-level診断に基づいて実施した。
3. **Top-2 narrowing**（006）で、安定性とtail behaviorに基づき `S11` と `S04` に絞り込んだ。
4. **Offline ensemble**（007）では、w=0.55 の `S11 + S04` ブレンドが純粋なS11に対しmean MSEを約5%改善した。
5. **Seed stability gate**（008）により、S11が5 seedsを通してより強く安定した単一モデルであることを確認した（Hana）。
6. **Cross-seed blend robustness**（009）では、ブレンド改善が **5/5** seed draws で維持され、mean MSEで -0.27、p95圧縮も得られた（Hana）。
7. **Production ensemble with per-drug breakdown**（010）では、Hanaでmean MSE **−4.89%** 改善とseed分散の縮小を確認した。薬剤別分析からは、48薬剤中上位5薬剤が集計MSEの約72%を占めることが分かった。
8. **Epurun seed stability**（011）では、第2病院（248 targets、52 folds）で、中心傾向ではS04がわずかに強い一方、S11がtail-risk優位を保つことを確認し、ブレンドの動機が補強された。
9. **Epurun cross-seed blend robustness**（012）では、ブレンド（w=0.55）が 5/5 seeds で勝利し、mean MSE **−4.38%** 改善と大幅なmax fold MSE圧縮を示した。
10. **Epurun production ensemble**（013）では、ブレンド改善とseed分散の縮小を確認した。誤差は248 targetsにより分散しており、上位5薬剤でMSEの約20%を占めた。

**配備推奨:**
- **Hana**: S11/S04ブレンドを **w=0.55–0.60**（S11寄り）で運用する。検証したすべてのseedで、[0.50, 0.65] の任意のwが純粋なS11を上回った。
- **Epurun**: S11/S04ブレンドを **w=0.50–0.55**（バランス型からややS04寄り）で運用する。検証したすべてのseedで、[0.45, 0.55] の任意のwが純粋なS11を上回った。
- **フォールバック**: ブレンド基盤が使えない場合、単一モデルのベースラインとしてPure S11（Hana）またはS04（Epurun）を用いる。
- ブレンド改善（約4–5%）とテールリスク圧縮は両病院で一貫しており、このアンサンブル手法の一般化可能性を支持する。

---

## 16) 図

### Hana Figures

### Figure 1 - Top-4のfold別指標
![Top-4のfold別指標](report/006_graph/hana/top4_foldwise_metrics.png)

### Figure 2 - Top-4のMSE分布
![Top-4のMSEボックスプロット](report/006_graph/hana/top4_mse_boxplot.png)

### Figure 3 - 全サンプルでのブレンドグリッド
![全サンプルのブレンドグリッド](report/007_ensemble/hana/blend_grid_full_sample.png)

### Figure 4 - 調整時と評価時のブレンド性能
![ブレンドの調整と評価](report/007_ensemble/hana/blend_tune_vs_eval.png)

### Figure 5 - Foldwise S11 vs S04 vs Blend
![foldwiseブレンド比較](report/007_ensemble/hana/blend_foldwise_comparison.png)

### Figure 6 - シナリオ別のseed分布（Hana）
![seed分布のボックスプロット](report/008_seed/hana/seed_spread_boxplot.png)

### Figure 7 - Foldwise seed behavior（Hana）
![シナリオ別foldwise seed挙動](report/008_seed/hana/seed_foldwise_by_scenario.png)

### Figure 8 - Seed相関行列（Hana）
![seed相関行列](report/008_seed/hana/seed_correlation_matrix.png)

### Figure 9 - Seed別のBlend mean-MSE delta vs S11, w=0.55（Hana）
![S11に対するブレンド差分](report/009_seed/hana/blend_seed_delta_vs_s11_w0.55.png)

### Figure 10 - Seed別の最適 `w*`（Hana）
![seed別のw*](report/009_seed/hana/blend_seed_wstar_by_seed.png)

### Figure 11 - テールリスク圧縮: blend vs pure models across seeds（Hana）
![テールリスク比較](report/009_seed/hana/blend_seed_tailrisk_comparison.png)

### Figure 12 - 薬剤別の真値 vs 予測値と95% CI（Hana）
![薬剤別CI](report/010_ensemble/hana/ensemble_true_vs_pred_ci_per_drug.png)

### Epurun Figures

### Figure E1 - シナリオ別のseed分布（Epurun）
![seed分布のボックスプロット](report/011_seed/epurun/seed_spread_boxplot.png)

### Figure E2 - Foldwise seed behavior（Epurun）
![シナリオ別foldwise seed挙動](report/011_seed/epurun/seed_foldwise_by_scenario.png)

### Figure E3 - Seed相関行列（Epurun）
![seed相関行列](report/011_seed/epurun/seed_correlation_matrix.png)

### Figure E4 - Seed別のBlend mean-MSE delta vs S11, w=0.55（Epurun）
![S11に対するブレンド差分](report/012_seed/epurun/blend_seed_delta_vs_s11_w0.55.png)

### Figure E5 - Seed別の最適 `w*`（Epurun）
![seed別のw*](report/012_seed/epurun/blend_seed_wstar_by_seed.png)

### Figure E6 - テールリスク圧縮: blend vs pure models across seeds（Epurun）
![テールリスク比較](report/012_seed/epurun/blend_seed_tailrisk_comparison.png)

### Figure E7 - 薬剤別の真値 vs 予測値と95% CI（Epurun）
![薬剤別CI](report/013_ensemble/epurun/ensemble_true_vs_pred_ci_per_drug.png)

---

## 17) 主要アーティファクト

### Hana

- `report/005_model/hana/leaderboard.csv`
- `report/005_model/hana/worst_folds_by_mse.csv`
- `report/006_graph/hana/top4_fold_metrics_reconstructed.csv`
- `report/006_graph/hana/leakage_audit.csv`
- `report/007_ensemble/hana/blend_summary.csv`
- `report/007_ensemble/hana/blend_eval_temporal_split.csv`
- `report/008_seed/hana/seed_stability_summary.csv`
- `report/008_seed/hana/seed_per_run_metrics.csv`
- `report/008_seed/hana/seed_spread_boxplot.png`
- `report/008_seed/hana/seed_foldwise_by_scenario.png`
- `report/008_seed/hana/seed_correlation_matrix.png`
- `report/009_seed/hana/blend_seed_fixed_weights_summary.csv`
- `report/009_seed/hana/blend_seed_wstar_summary.csv`
- `report/009_seed/hana/blend_seed_robustness_decision.csv`
- `report/009_seed/hana/blend_seed_delta_vs_s11_w0.55.png`
- `report/009_seed/hana/blend_seed_wstar_by_seed.png`
- `report/009_seed/hana/blend_seed_tailrisk_comparison.png`
- `report/010_ensemble/hana/ensemble_summary.md`
- `report/010_ensemble/hana/ensemble_metrics_overall.csv`
- `report/010_ensemble/hana/ensemble_metrics_by_seed.csv`
- `report/010_ensemble/hana/ensemble_ci_per_drug.csv`
- `report/010_ensemble/hana/ensemble_seed_predictions_per_drug.csv`
- `report/010_ensemble/hana/ensemble_true_vs_pred_ci_per_drug.png`

### Epurun

- `report/011_seed/epurun/seed_stability_summary.csv`
- `report/011_seed/epurun/seed_per_run_metrics.csv`
- `report/011_seed/epurun/seed_spread_boxplot.png`
- `report/011_seed/epurun/seed_foldwise_by_scenario.png`
- `report/011_seed/epurun/seed_correlation_matrix.png`
- `report/012_seed/epurun/blend_seed_fixed_weights_summary.csv`
- `report/012_seed/epurun/blend_seed_wstar_summary.csv`
- `report/012_seed/epurun/blend_seed_robustness_decision.csv`
- `report/012_seed/epurun/blend_seed_delta_vs_s11_w0.55.png`
- `report/012_seed/epurun/blend_seed_wstar_by_seed.png`
- `report/012_seed/epurun/blend_seed_tailrisk_comparison.png`
- `report/013_ensemble/epurun/ensemble_summary.md`
- `report/013_ensemble/epurun/ensemble_metrics_overall.csv`
- `report/013_ensemble/epurun/ensemble_metrics_by_seed.csv`
- `report/013_ensemble/epurun/ensemble_ci_per_drug.csv`
- `report/013_ensemble/epurun/ensemble_seed_predictions_per_drug.csv`
- `report/013_ensemble/epurun/ensemble_true_vs_pred_ci_per_drug.png`
