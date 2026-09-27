#!/usr/bin/env python3
"""Minimal RuleKAN package smoke test.

This example uses only the dependencies of the installable ``rulekan`` package.
It does not import the benchmark harness or any external symbolic-regression
baseline such as PySR, SR-KAN, or Operon.
"""

from __future__ import annotations

import torch

import rulekan
from rulekan import SumProductKAN


def main() -> None:
    torch.manual_seed(0)

    # Small synthetic regression problem with a multiplicative interaction.
    x = 2.0 * torch.rand(128, 2) - 1.0
    y = 0.75 * x[:, [0]] * x[:, [1]] + 0.20 * x[:, [0]]

    model = SumProductKAN(
        in_dim=2,
        n_rules=3,
        max_factors=2,
        grid=4,
        k=2,
        symbolic_library=("x", "sin"),
        seed=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    initial_loss = None

    for _ in range(25):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during RuleKAN smoke test")
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(torch.mean((model(x) - y) ** 2))
        sample = model(x[:4])

    if sample.shape != (4, 1):
        raise RuntimeError(f"unexpected output shape: {tuple(sample.shape)}")
    if not torch.isfinite(sample).all():
        raise RuntimeError("non-finite RuleKAN predictions")
    if initial_loss is None or final_loss >= initial_loss:
        raise RuntimeError(
            f"training did not reduce MSE: initial={initial_loss}, final={final_loss}"
        )

    print(f"rulekan version: {rulekan.__version__}")
    print(f"initial MSE: {initial_loss:.6f}")
    print(f"final MSE:   {final_loss:.6f}")
    print("RuleKAN package smoke test: OK")


if __name__ == "__main__":
    main()
