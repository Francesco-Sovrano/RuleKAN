from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans


class CompactANFIS(torch.nn.Module):
    """Compact first-order Takagi-Sugeno ANFIS regression model.

    Each rule uses a product of Gaussian antecedent memberships over all input
    variables.  Products are evaluated in log space and normalized with a
    softmax.  Each consequent is affine in the inputs.  Premise parameters are
    optimized with gradients; consequent coefficients are refit by ridge least
    squares, following the hybrid-learning idea of ANFIS while keeping the
    implementation stable for the benchmark's moderate dimensions.
    """

    def __init__(
        self,
        in_dim: int,
        n_rules: int,
        *,
        min_sigma: float = 0.05,
        max_sigma: float = 5.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("ANFIS requires at least one input variable")
        if n_rules < 1:
            raise ValueError("ANFIS requires at least one fuzzy rule")
        if not (0 < float(min_sigma) < float(max_sigma)):
            raise ValueError("ANFIS requires 0 < min_sigma < max_sigma")
        self.in_dim = int(in_dim)
        self.n_rules = int(n_rules)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.centers = torch.nn.Parameter(
            torch.zeros(self.n_rules, self.in_dim, dtype=dtype, device=device)
        )
        self.log_sigma = torch.nn.Parameter(
            torch.zeros(self.n_rules, self.in_dim, dtype=dtype, device=device)
        )
        # Consequents are solved in closed form.  Keeping them as a parameter
        # makes state_dict/checkpointing simple while requires_grad=False keeps
        # Adam restricted to the premise parameters.
        self.consequents = torch.nn.Parameter(
            torch.zeros(self.n_rules, self.in_dim + 1, dtype=dtype, device=device),
            requires_grad=False,
        )

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)

    def initialize_from_data(self, x: torch.Tensor, *, seed: int = 0, n_init: int = 10) -> None:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"expected x with shape [N,{self.in_dim}], got {tuple(x.shape)}")
        xx = np.asarray(x.detach().cpu(), dtype=np.float64)
        k = min(self.n_rules, max(1, int(xx.shape[0])))
        if k == 1:
            centers = xx.mean(axis=0, keepdims=True)
            labels = np.zeros(xx.shape[0], dtype=int)
        else:
            km = KMeans(n_clusters=k, n_init=max(1, int(n_init)), random_state=int(seed))
            labels = km.fit_predict(xx)
            centers = np.asarray(km.cluster_centers_, dtype=np.float64)
        if k < self.n_rules:
            # Deterministic padding is only relevant for tiny smoke datasets.
            pad = np.repeat(centers[-1:, :], self.n_rules - k, axis=0)
            centers = np.concatenate([centers, pad], axis=0)

        global_std = np.std(xx, axis=0, ddof=0)
        global_std = np.maximum(global_std, 0.25)
        sigmas = np.empty((self.n_rules, self.in_dim), dtype=np.float64)
        for r in range(self.n_rules):
            if r < k:
                pts = xx[labels == r]
                if pts.shape[0] >= 2:
                    local = np.std(pts, axis=0, ddof=0)
                    sigmas[r] = np.where(local > 0.08, local, global_std)
                else:
                    sigmas[r] = global_std
            else:
                sigmas[r] = global_std
        sigmas = np.clip(sigmas * 1.5, self.min_sigma, self.max_sigma)
        with torch.no_grad():
            self.centers.copy_(torch.as_tensor(centers, dtype=self.centers.dtype, device=self.centers.device))
            self.log_sigma.copy_(
                torch.log(torch.as_tensor(sigmas, dtype=self.log_sigma.dtype, device=self.log_sigma.device))
            )
            self.consequents.zero_()

    def log_firing(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma
        z = (x[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]
        return -0.5 * torch.sum(z.square(), dim=-1)

    def firing(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.log_firing(x), dim=1)

    def rule_outputs(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.cat((torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x), dim=1)
        return phi @ self.consequents.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.firing(x)
        consequents = self.rule_outputs(x)
        return torch.sum(weights * consequents, dim=1, keepdim=True)

    def refit_consequents(self, x: torch.Tensor, y: torch.Tensor, *, ridge: float = 1e-4) -> None:
        """Closed-form ridge fit of all first-order Sugeno consequents."""
        with torch.no_grad():
            weights = self.firing(x).to(torch.float64)
            phi = torch.cat(
                (torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x), dim=1
            ).to(torch.float64)
            design = (weights[:, :, None] * phi[:, None, :]).reshape(x.shape[0], -1)
            yy = y.reshape(y.shape[0], -1).mean(dim=1, keepdim=True).to(torch.float64)
            p = design.shape[1]
            gram = design.T @ design
            rhs = design.T @ yy
            eye = torch.eye(p, dtype=torch.float64, device=x.device)
            try:
                beta = torch.linalg.solve(gram + float(ridge) * eye, rhs)[:, 0]
            except Exception:
                aug_x = torch.cat((design, math.sqrt(max(float(ridge), 0.0)) * eye), dim=0)
                aug_y = torch.cat((yy, torch.zeros(p, 1, dtype=torch.float64, device=x.device)), dim=0)
                beta = torch.linalg.lstsq(aug_x, aug_y).solution[:, 0]
            beta = beta.reshape(self.n_rules, self.in_dim + 1)
            self.consequents.copy_(beta.to(self.consequents.dtype))

    def premise_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            w = self.firing(x)
            usage = w.mean(dim=0)
            entropy = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(dim=1).mean()
            effective = int((usage >= (0.01 / max(1, self.n_rules))).sum().item())
            return {
                "anfis_effective_rules": float(effective),
                "anfis_firing_entropy": float(entropy.detach().cpu()),
                "anfis_min_sigma": float(self.sigma.min().detach().cpu()),
                "anfis_max_sigma": float(self.sigma.max().detach().cpu()),
                "anfis_max_rule_usage": float(usage.max().detach().cpu()),
            }


@dataclass
class ANFISFitResult:
    model: CompactANFIS
    best_val_mse: float
    epochs_run: int
    seconds: float


def fit_compact_anfis(
    model: CompactANFIS,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    epochs: int = 120,
    lr: float = 2e-2,
    ridge: float = 1e-4,
    patience: int = 20,
    min_delta_rel: float = 1e-5,
    premise_l2: float = 1e-5,
) -> ANFISFitResult:
    """Hybrid ANFIS fit with validation checkpointing.

    Training coordinates fit premise and consequent parameters. Validation data
    select the checkpoint. Test data are intentionally absent from this API.
    """
    optimizer = torch.optim.Adam([model.centers, model.log_sigma], lr=float(lr))
    best_state: Optional[dict] = None
    best_val = float("inf")
    stale = 0
    t0 = time.perf_counter()
    n_epochs = max(1, int(epochs))
    epochs_run = 0
    for epoch in range(n_epochs):
        model.train()
        model.refit_consequents(train_x, train_y, ridge=float(ridge))
        optimizer.zero_grad(set_to_none=True)
        pred = model(train_x)
        loss = torch.mean((pred - train_y) ** 2)
        if premise_l2 > 0:
            loss = loss + float(premise_l2) * model.log_sigma.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.log_sigma.clamp_(math.log(model.min_sigma), math.log(model.max_sigma))
        model.refit_consequents(train_x, train_y, ridge=float(ridge))
        model.eval()
        with torch.no_grad():
            vmse = float(torch.mean((model(val_x) - val_y) ** 2).detach().cpu())
        epochs_run = epoch + 1
        improved = math.isfinite(vmse) and (
            not math.isfinite(best_val)
            or vmse < best_val * (1.0 - max(0.0, float(min_delta_rel)))
        )
        if improved:
            best_val = vmse
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= max(1, int(patience)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return ANFISFitResult(
        model=model,
        best_val_mse=float(best_val),
        epochs_run=int(epochs_run),
        seconds=float(time.perf_counter() - t0),
    )
