# LAVAR: 잠재 오토인코더 VAR 예측 패키지

언어: [English](LAVAR_README.md) | [한국어](LAVAR_README_KOR.md) | [日本語](LAVAR_README_JPN.md)

[![paper](https://img.shields.io/badge/paper-TBD-lightgrey)](#citation)
[![python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](#installation)
[![pytorch](https://img.shields.io/badge/PyTorch-%3E%3D2.9.1-ee4c2c)](https://pytorch.org/)
[![pydantic](https://img.shields.io/badge/Pydantic-%3E%3D2.0-e92063)](https://docs.pydantic.dev/)
[![uv](https://img.shields.io/badge/uv-locked-6f42c1)](https://github.com/astral-sh/uv)

`lavar`는 다단계 의료 공급 예측을 위한 간결한 연구용 PyTorch 패키지입니다.
패키지는 2단계 파이프라인으로 구성됩니다.

1. Stage 1은 비공급 공변량으로부터 잠재 표현과 잠재 동역학을 학습합니다.
2. Stage 2는 잠재 백본을 고정한 뒤, 미래 공급 타깃에 대해 밀도 분할된 공급 헤드를 학습합니다.

## Public API

```python
from lavar import LAVARForecaster, LAVARConfig, RollingEvaluator, EvaluationResults
```

주요 진입점은 다음과 같습니다.

- `LAVARConfig`: 모델, 학습, 추론 동작을 정의하는 pydantic 설정
- `LAVARForecaster`: `fit()`, `fit_heads()`, `predict()`, `save()`, `load()`를 제공하는 sklearn 스타일 래퍼
- `RollingEvaluator`: rolling-origin 평가 유틸리티
- `EvaluationResults`: `to_dataframe()`와 `plot()`를 제공하는 결과 객체

## 현재 아키텍처

### Stage 1 백본

- 관측 모델: MLP encoder + MLP decoder
- 잠재 동역학: `VAR` 또는 `GRU` 선택 가능
- 선택적 입력 확장: `stage1_use_supply_history=True`이면 Stage 1 입력에 공급 이력을 연결

### Stage 2 공급 모델링

- Stage 2 모드:
  - `baseline`: 잠재 rollout만 사용
  - `supply_history_latent`: 공급 이력이 포함된 encoder 입력과 잠재 rollout을 함께 사용
- 타깃 분할: 타깃별 nonzero rate 기준으로 dense / sparse / ultra-sparse 버킷 구성
- 지원하는 헤드 계열:
  - MLP 기반 deterministic delta 회귀
  - GRU 기반 deterministic delta 회귀
  - 확률적 NB 헤드
  - 확률적 ZINB 헤드

### 현재 기본 파이프라인 매핑

현재 `LAVARForecaster`의 density-split Stage 2 학습 파이프라인은 다음과 같이 동작합니다.

| 버킷 | 기본 head type | 비고 |
| --- | --- | --- |
| Dense | `delta_mse` | deterministic delta 예측 |
| Sparse | `delta_mse` | deterministic delta 예측 |
| Ultra-sparse | `zinb` | 확률적 count head |

`nb`는 저수준 Stage 2 모델 및 trainer API에서는 지원되지만, 기본 dispatcher는 현재 ultra-sparse 버킷을 `zinb`로 라우팅합니다.

## 지원 Head Matrix

| Head 옵션 | 클래스 경로 | 백엔드 | 출력 | 현재 사용 방식 |
| --- | --- | --- | --- | --- |
| `delta_mse` | `SupplyHeadMSE` | MLP | step별 `delta` | dense/sparse 기본 헤드 |
| `delta_mse` + `stage2_head_type="gru"` | `LAVARWithSupplyGRU`를 통한 `SupplyHeadGRU` | GRU | 시퀀스 `delta` 값 | baseline 모드 delta head에서 사용 가능 |
| `nb` | `SupplyHeadNB` | MLP | `mu`, `theta` | 저수준 옵션으로 지원 |
| `zinb` | `SupplyHeadZINB` | MLP | `pi`, `mu`, `theta` | ultra-sparse 기본 헤드 |

구현상 중요한 점:

- `stage2_head_type="gru"`는 `delta_mse` head에만 적용됩니다.
- `LAVARWithSupplyGRU`는 deterministic delta head 전용입니다.
- NB와 ZINB head는 계속 pointwise MLP head로 유지됩니다.
- `supply_history_latent`를 사용하려면 `use_supply_history=True`와 `stage1_use_supply_history=True`가 모두 필요합니다.

## 학습 흐름

1. `fit_stage1_shared(X, y)`
   - `X`를 스케일링
   - Stage 1이 공급 이력을 사용할 때만 `y`도 스케일링
   - reconstruction loss와 latent dynamics loss로 `LAVAR` 학습
   - optional multi-step latent supervision은 latent rollout과 미래 observation의 encoded latent를 비교
2. `fit_stage2_private(X, y)`
   - Stage 1 가중치를 고정
   - 학습 윈도우에서만 타깃 density bucket 계산
   - 버킷별 supply head 학습
3. `predict(X_recent, y_recent)`
   - 미래 latent state rollout 수행
   - 학습된 supply head를 버킷별로 적용
   - prediction guardrail, optional zero gating, optional integer rounding 적용

delta head의 경우 Stage 2는 raw increment를 예측하고 마지막 관측 공급값 `y0`에서부터 적분합니다.
확률적 head의 경우 기본 점예측은 예측된 평균 `mu`입니다.

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

builder 스타일 설정도 지원합니다.

```python
cfg = (
    LAVARConfig.builder()
    .device("cpu")
    .latent(dim=8, encoder=[32, 16], decoder=[16, 32])
    .horizon(h=14, history=7)
    .build()
)
```

## 주요 설정 항목

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| `device` | `"mps"` | 학습/추론 디바이스 |
| `dyn_p` | `7` | 잠재 이력 길이 / 동역학 차수 |
| `horizon` | `14` | 예측 horizon |
| `latent_dim` | `8` | 잠재 상태 차원 |
| `latent_dynamics_type` | `"var"` | Stage 1 잠재 동역학: `var` 또는 `gru` |
| `use_supply_history` | `False` | Stage 2 encoder 입력에 공급 이력 연결 |
| `stage1_use_supply_history` | `False` | Stage 1 학습 시 공급 이력 연결 |
| `stage2_mode` | `"baseline"` | Stage 2 학습 모드 |
| `stage2_head_type` | `"mlp"` | baseline 모드 delta head 백엔드: `mlp` 또는 `gru` |
| `dense_nonzero_rate_thr` | `0.70` | dense bucket 임계값 |
| `ultra_nonzero_rate_thr` | `0.005` | ultra-sparse bucket 임계값 |
| `horizon_loss_weight` | `"uniform"` | baseline 모드 delta loss 가중 방식 |
| `stage2_delta_nonneg_mode` | `"clamp"` | 적분된 delta forecast의 음수 방지 방식 |
| `pred_guardrail_quantile` | `0.995` | prediction clipping 상한 quantile |
| `zero_gate_k` | `7` | zero gating lookback window |
| `forecast_round_to_int` | `True` | 추론 시 정수 반올림 여부 |

`stage2_use_explicit_lag_coeff`는 설정에 존재하지만 현재 패키지에서 사용되지 않는 예약 실험 플래그입니다.

## 데이터 계약과 Shape

학습 입력 시계열은 정렬되어 있어야 합니다.

- `X`: shape `(T, Dx)`
- `y`: shape `(T, Dy)`

공개 메서드 shape:

- `fit(X, y)`: `X: (T, Dx)`, `y: (T, Dy)`
- `fit_heads(X, y)`: `fit()`와 동일
- `predict(X_recent, y_recent)`:
  - `X_recent`: `(dyn_p, Dx)` 또는 `(dyn_p + 1, Dx)`
  - `y_recent`: supply history를 사용할 때는 같은 길이가 필요하며, 그렇지 않더라도 delta head 적분 기준값으로 최근 공급값을 쓰기 때문에 전달을 권장
  - 반환값: shape `(horizon, Dy)`의 `np.ndarray`

윈도우 데이터셋:

- `RollingXYDataset`: `(x_past, x_future, y_future)`
- `RollingXYDatasetWithY0`: `(x_past, x_future, y0, y_future)`

`use_supply_history=True`이면 데이터셋은 `x_past`, `x_future`에 `y` 컬럼을 이어 붙여 encoder가 `Dx + Dy` 차원을 보게 됩니다.

## 패키지 레이아웃

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

저장소 루트에서 실행합니다.

### Option A: `uv` (권장)

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

## 체크포인트와 출력물

기본 학습 헬퍼는 save path가 주어졌을 때 다음 체크포인트 이름을 사용합니다.

- `lavar_best.pth`
- `lavar_supply_dense_best.pth`
- `lavar_supply_sparse_best.pth`
- `lavar_supply_ultra_best.pth`

`LAVARForecaster.save(path)`는 config, scaler, Stage 1 가중치, Stage 2 head 가중치, bucket index, prediction guardrail bound를 하나의 아티팩트로 저장합니다.

## 참고 제약 사항

- `latent_dim`을 바꾸면 두 Stage 모두 다시 학습해야 합니다.
- `dyn_p`를 바꾸면 latent dynamics shape가 바뀌므로 다시 학습해야 합니다.
- Stage 2 head 학습 중에는 Stage 1 `LAVAR` 가중치를 풀지 마세요.
- density bucket index는 저장된 head 가중치와 함께 동작하는 Stage 2 계약의 일부입니다.

## Citation

이 패키지를 사용한다면 관련 논문을 인용해 주세요.

```bibtex
@article{lavar_tbd,
  title   = {LAVAR: Latent Autoencoder VAR for Forecasting},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## Acknowledgements

PyTorch와 pydantic 기반으로 작성되었으며, rolling healthcare demand / supply forecasting 실험을 위해 코드 표면을 의도적으로 작게 유지했습니다.
