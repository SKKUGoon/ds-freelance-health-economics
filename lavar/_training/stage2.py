"""Stage 2 dispatcher — density-split supply-head training.

Shared utilities live here; mode-specific training loops live in:
  - ts_latent.py            (stage2_mode="ts_latent", default)
  - ts_supply_ts_latent.py  (stage2_mode="ts_supply_ts_latent")
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from lavar.config import LAVARConfig
from lavar._core.model import LAVAR

from lavar._training.ts_latent import (
    train_supply_head_indexed as _train_head_ts_latent,
)
from lavar._training.ts_supply_ts_latent import (
    train_supply_head_indexed as _train_head_ts_supply,
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_nonzero_rate_from_loader(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute per-target nonzero rate from a train DataLoader.

    Expected batch: (x_past, x_future, y_future) where y_future has shape (B, H, Dy).
    Returns: nonzero_rate with shape (Dy,), on CPU.
    """
    nonzero_sum: Optional[torch.Tensor] = None
    count_sum: int = 0

    for batch in train_loader:
        if len(batch) == 3:
            _x_past, _x_future, y_future = batch
        elif len(batch) == 4:
            _x_past, _x_future, _y0, y_future = batch
        else:
            raise ValueError(
                "Density split training expects batches of (x_past, x_future, y_future). "
                f"Got batch size {len(batch)}."
            )
        # y_future: (B, H, Dy)
        y_flat = y_future.reshape(-1, y_future.shape[-1])  # (B*H, Dy)
        nz = (y_flat > 0).to(dtype=torch.float32).sum(dim=0)  # (Dy,)
        if nonzero_sum is None:
            nonzero_sum = nz.detach().cpu()
        else:
            nonzero_sum += nz.detach().cpu()
        count_sum += int(y_flat.shape[0])

    if nonzero_sum is None or count_sum == 0:
        raise RuntimeError(
            "train_loader yielded no batches; cannot compute nonzero rates."
        )

    return nonzero_sum / float(count_sum)  # (Dy,)


def split_indices_by_density(
    nonzero_rate: torch.Tensor,
    dense_thr: float,
    ultra_thr: float,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split Dy indices into dense/sparse/ultra-sparse based on nonzero_rate thresholds.
    """
    if nonzero_rate.dim() != 1:
        raise ValueError(
            f"nonzero_rate must have shape (Dy,). Got {tuple(nonzero_rate.shape)}"
        )
    if not (0.0 <= ultra_thr <= dense_thr <= 1.0):
        raise ValueError(
            f"Expected 0 <= ultra_thr <= dense_thr <= 1. Got ultra={ultra_thr}, dense={dense_thr}"
        )

    dense_mask = nonzero_rate >= dense_thr
    ultra_mask = nonzero_rate <= ultra_thr
    sparse_mask = (~dense_mask) & (~ultra_mask)

    dense_idx = dense_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    sparse_idx = sparse_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    ultra_idx = ultra_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    return dense_idx, sparse_idx, ultra_idx


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_MODE_DISPATCH = {
    "ts_latent": _train_head_ts_latent,
    "ts_supply_ts_latent": _train_head_ts_supply,
}


def train_supply_heads(
    lavar: LAVAR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LAVARConfig,
    save_dense: Optional[str] = "lavar_supply_dense_best.pth",
    save_sparse: Optional[str] = "lavar_supply_sparse_best.pth",
    save_ultra: Optional[str] = "lavar_supply_ultra_best.pth",
) -> Dict[str, object]:
    """
    Stage 2: train three supply heads split by value density (nonzero rate).

      - dense bucket  -> delta-MSE head
      - sparse bucket -> delta-MSE head
      - ultra-sparse  -> ZINB head

    Dispatches to the mode-specific training loop based on cfg.stage2_mode.
    """
    mode = cfg.stage2_mode
    if mode not in _MODE_DISPATCH:
        raise ValueError(
            f"Unknown stage2_mode={mode!r}. Expected one of {sorted(_MODE_DISPATCH)}."
        )

    _train_head = _MODE_DISPATCH[mode]

    # --- density split ---
    nonzero_rate = compute_nonzero_rate_from_loader(train_loader)
    dense_thr = float(cfg.dense_nonzero_rate_thr)
    ultra_thr = float(cfg.ultra_nonzero_rate_thr)
    dense_idx, sparse_idx, ultra_idx = split_indices_by_density(
        nonzero_rate, dense_thr=dense_thr, ultra_thr=ultra_thr
    )

    out: Dict[str, object] = {
        "dense_indices": dense_idx,
        "sparse_indices": sparse_idx,
        "ultra_indices": ultra_idx,
        "dense_model": None,
        "sparse_model": None,
        "ultra_model": None,
    }

    out["dense_model"] = _train_head(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=dense_idx,
        head_type="delta_mse",
        save_path=save_dense,
    )
    out["sparse_model"] = _train_head(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=sparse_idx,
        head_type="delta_mse",
        save_path=save_sparse,
    )
    out["ultra_model"] = _train_head(
        lavar=lavar,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        y_indices=ultra_idx,
        head_type="zinb",
        save_path=save_ultra,
    )

    return out
