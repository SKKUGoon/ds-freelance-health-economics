# LAVAR: 潜在オートエンコーダVAR予測パッケージ

言語: [English](LAVAR_README.md) | [한국어](LAVAR_README_KOR.md) | [日本語](LAVAR_README_JPN.md)

[![paper](https://img.shields.io/badge/paper-TBD-lightgrey)](#citation)
[![python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](#installation)
[![pytorch](https://img.shields.io/badge/PyTorch-%3E%3D2.9.1-ee4c2c)](https://pytorch.org/)
[![pydantic](https://img.shields.io/badge/Pydantic-%3E%3D2.0-e92063)](https://docs.pydantic.dev/)
[![uv](https://img.shields.io/badge/uv-locked-6f42c1)](https://github.com/astral-sh/uv)

`lavar` は、マルチステップの医療供給予測向けに設計された、コンパクトな研究用PyTorchパッケージです。
パッケージは2段階のパイプラインを実装しています。

1. Stage 1で、非供給共変量から潜在表現と潜在ダイナミクスを学習します。
2. Stage 2で、潜在バックボーンを凍結し、将来の供給ターゲットに対して密度分割された供給ヘッドを学習します。

## Public API

```python
from lavar import LAVARForecaster, LAVARConfig, RollingEvaluator, EvaluationResults
```

主な公開エントリポイントは次のとおりです。

- `LAVARConfig`: モデル、学習、推論動作を定義するpydantic設定
- `LAVARForecaster`: `fit()`, `fit_heads()`, `predict()`, `save()`, `load()` を提供するsklearn風ラッパー
- `RollingEvaluator`: rolling-origin評価ユーティリティ
- `EvaluationResults`: `to_dataframe()` と `plot()` を持つ結果オブジェクト

## 現在のアーキテクチャ

### Stage 1バックボーン

- 観測モデル: MLP encoder + MLP decoder
- 潜在ダイナミクス: `VAR` または `GRU`
- オプションの入力拡張: `stage1_use_supply_history=True` のとき、Stage 1入力に供給履歴を連結

### Stage 2供給モデリング

- Stage 2モード:
  - `baseline`: 潜在rolloutのみ
  - `supply_history_latent`: 供給履歴を含むencoder入力と潜在rolloutを併用
- ターゲット分割: ターゲットごとのnonzero rateに基づく dense / sparse / ultra-sparse バケット
- サポートされるヘッド族:
  - MLPベースのdeterministic delta回帰
  - GRUベースのdeterministic delta回帰
  - 確率的NBヘッド
  - 確率的ZINBヘッド

### 現在のデフォルトパイプライン対応

現在の `LAVARForecaster` の density-split Stage 2 学習パイプラインは次の対応を使います。

| バケット | デフォルトhead type | 備考 |
| --- | --- | --- |
| Dense | `delta_mse` | deterministic delta予測 |
| Sparse | `delta_mse` | deterministic delta予測 |
| Ultra-sparse | `zinb` | 確率的count head |

`nb` は低レベルのStage 2モデルおよびtrainer APIで利用可能ですが、デフォルトdispatcherは現在ultra-sparseバケットを `zinb` に割り当てます。

## サポートHead Matrix

| Headオプション | クラスパス | バックエンド | 出力 | 現在の使われ方 |
| --- | --- | --- | --- | --- |
| `delta_mse` | `SupplyHeadMSE` | MLP | 各stepの `delta` | dense/sparseのデフォルトヘッド |
| `delta_mse` + `stage2_head_type="gru"` | `LAVARWithSupplyGRU` 経由の `SupplyHeadGRU` | GRU | `delta` の系列 | baselineモードのdelta headで利用可能 |
| `nb` | `SupplyHeadNB` | MLP | `mu`, `theta` | 低レベルオプションとしてサポート |
| `zinb` | `SupplyHeadZINB` | MLP | `pi`, `mu`, `theta` | ultra-sparseのデフォルトヘッド |

実装上の重要事項:

- `stage2_head_type="gru"` は `delta_mse` ヘッドにのみ適用されます。
- `LAVARWithSupplyGRU` はdeterministic deltaヘッド専用です。
- NBとZINBヘッドは引き続きpointwise MLPヘッドです。
- `supply_history_latent` を使うには `use_supply_history=True` と `stage1_use_supply_history=True` の両方が必要です。

## 学習フロー

1. `fit_stage1_shared(X, y)`
   - `X` をスケーリング
   - Stage 1が供給履歴を使う場合のみ `y` もスケーリング
   - reconstruction loss と latent dynamics loss で `LAVAR` を学習
   - optional multi-step latent supervision では latent rollout と将来観測のencoded latentを比較
2. `fit_stage2_private(X, y)`
   - Stage 1重みを凍結
   - 学習ウィンドウ上のみでターゲット密度バケットを計算
   - バケットごとにsupply headを学習
3. `predict(X_recent, y_recent)`
   - 将来のlatent stateをrollout
   - 学習済みsupply headをバケット単位で適用
   - prediction guardrail、optional zero gating、optional integer roundingを適用

deltaヘッドでは、Stage 2はraw incrementを予測し、最後に観測した供給値 `y0` から積み上げます。
確率的ヘッドでは、デフォルトの点予測は予測平均 `mu` です。

## Quickstart

```python
import numpy as np

from lavar import LAVARConfig, LAVARForecaster, RollingEvaluator

cfg = LAVARConfig(
    device="cpu",
    dyn_p=7,
    horizon=14,
    latent_dim=8,
    latent_dynamics_type="gru",
    stage2_mode="baseline",
    stage2_head_type="gru",
    epochs_lavar=5,
    epochs_supply=5,
)

T, Dx, Dy = 512, 20, 6
X = np.random.randn(T, Dx).astype("float32")
y = np.abs(np.random.randn(T, Dy).astype("float32"))

model = LAVARForecaster(cfg)
model.fit(X, y)

X_recent = X[-cfg.dyn_p :]
y_recent = y[-cfg.dyn_p :]
forecast = model.predict(X_recent, y_recent=y_recent)

model.save("lavar_model.pth")
loaded = LAVARForecaster.load("lavar_model.pth")

results = RollingEvaluator(cfg).evaluate(X, y, fold_step=14, verbose=False)
print(results.summary)
```

builderスタイルの設定も利用できます。

```python
cfg = (
    LAVARConfig.builder()
    .device("cpu")
    .latent(dim=8, encoder=[32, 16], decoder=[16, 32])
    .horizon(h=14, history=7)
    .build()
)
```

## 主な設定項目

| フィールド | デフォルト | 意味 |
| --- | --- | --- |
| `device` | `"mps"` | 学習/推論デバイス |
| `dyn_p` | `7` | 潜在履歴長 / ダイナミクス次数 |
| `horizon` | `14` | 予測horizon |
| `latent_dim` | `8` | 潜在状態次元 |
| `latent_dynamics_type` | `"var"` | Stage 1潜在ダイナミクス: `var` または `gru` |
| `use_supply_history` | `False` | Stage 2 encoder入力に供給履歴を連結 |
| `stage1_use_supply_history` | `False` | Stage 1学習時に供給履歴を連結 |
| `stage2_mode` | `"baseline"` | Stage 2学習モード |
| `stage2_head_type` | `"mlp"` | baselineモードdelta headバックエンド: `mlp` または `gru` |
| `dense_nonzero_rate_thr` | `0.70` | denseバケット閾値 |
| `ultra_nonzero_rate_thr` | `0.005` | ultra-sparseバケット閾値 |
| `horizon_loss_weight` | `"uniform"` | baselineモードdelta lossの重み付け |
| `stage2_delta_nonneg_mode` | `"clamp"` | 積分後delta予測の非負射影方法 |
| `pred_guardrail_quantile` | `0.995` | prediction clipping上限のquantile |
| `zero_gate_k` | `7` | zero gatingのlookback window |
| `forecast_round_to_int` | `True` | 推論時に整数へ丸めるか |

`stage2_use_explicit_lag_coeff` は設定上は存在しますが、現在のパッケージでは未使用の予約実験フラグです。

## データ契約とShape

学習では、位置合わせ済みの時系列を想定します。

- `X`: shape `(T, Dx)`
- `y`: shape `(T, Dy)`

公開メソッドのshape:

- `fit(X, y)`: `X: (T, Dx)`, `y: (T, Dy)`
- `fit_heads(X, y)`: `fit()` と同じ
- `predict(X_recent, y_recent)`:
  - `X_recent`: `(dyn_p, Dx)` または `(dyn_p + 1, Dx)`
  - `y_recent`: supply historyを使う場合は同じ時間長が必要で、それ以外でもdelta headの積分基準として最新供給値を使うため、渡すことを推奨
  - 戻り値: shape `(horizon, Dy)` の `np.ndarray`

ウィンドウ用データセット:

- `RollingXYDataset`: `(x_past, x_future, y_future)`
- `RollingXYDatasetWithY0`: `(x_past, x_future, y0, y_future)`

`use_supply_history=True` の場合、データセットは `x_past` と `x_future` に `y` 列を連結し、encoderは `Dx + Dy` 次元を受け取ります。

## パッケージ構成

```text
lavar/
├── __init__.py
├── config.py
├── forecaster.py
├── evaluation.py
├── losses.py
├── _core/
│   ├── dynamics.py
│   ├── heads.py
│   └── model.py
├── _data/
│   ├── dataset.py
│   └── scaler.py
└── _training/
    ├── stage1.py
    └── stage2/
        ├── __init__.py
        ├── common.py
        ├── stage2_test_baseline.py
        └── stage2_test_supply_history_latent.py
```

## Installation

リポジトリルートから実行します。

### Option A: `uv`（推奨）

```bash
uv sync
```

### Option B: `pip` editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## チェックポイントと出力

デフォルトの学習ヘルパーは、save pathが与えられた場合に次のチェックポイント名を使います。

- `lavar_best.pth`
- `lavar_supply_dense_best.pth`
- `lavar_supply_sparse_best.pth`
- `lavar_supply_ultra_best.pth`

`LAVARForecaster.save(path)` は、config、scaler、Stage 1重み、Stage 2 head重み、bucket index、prediction guardrail bound を単一アーティファクトに保存します。

## 制約と注意事項

- `latent_dim` を変更した場合は両Stageを再学習してください。
- `dyn_p` を変更するとlatent dynamicsのshapeが変わるため再学習が必要です。
- Stage 2 head学習中にStage 1 `LAVAR` 重みを解凍しないでください。
- density bucket indexは保存されたhead重みと整合するStage 2契約の一部です。

## Citation

このパッケージを利用する場合は、関連論文を引用してください。

```bibtex
@article{lavar_tbd,
  title   = {LAVAR: Latent Autoencoder VAR for Forecasting},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## Acknowledgements

PyTorchとpydanticの上に構築され、rolling healthcare demand / supply forecasting実験のために意図的に小さなコード面積を保っています。
