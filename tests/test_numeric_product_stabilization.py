import torch

from symbolic_kan.sum_product_kan import SumProductKAN


def _model(factor_scale: float, product_scale: float) -> SumProductKAN:
    return SumProductKAN(
        in_dim=2,
        n_rules=2,
        max_factors=2,
        grid=4,
        k=2,
        numeric_factor_gradient_scale=factor_scale,
        numeric_product_gradient_scale=product_scale,
        seed=7,
    )


def test_numeric_product_stabilization_is_exact_forward_and_reduces_large_gradients():
    plain = _model(0.0, 0.0)
    stable = _model(3.0, 2.0)
    stable.load_state_dict(plain.state_dict())
    with torch.no_grad():
        # Deliberately make product-rule contributions large enough to stress
        # the gradient geometry without changing either model's represented function.
        plain.rule_scale.fill_(80.0)
        stable.rule_scale.fill_(80.0)

    x = torch.tensor(
        [[-1.5, -1.0], [-0.5, 0.7], [0.3, -0.2], [1.2, 1.3]],
        dtype=torch.float32,
    )
    for model in (plain, stable):
        model.train()
        model.symbolic_enabled = False
        model.zero_grad(set_to_none=True)

    y_plain = plain(x)
    y_stable = stable(x)
    # The stabilization is a custom-autograd identity: only backward changes.
    assert torch.equal(y_plain, y_stable)

    y_plain.square().mean().backward()
    y_stable.square().mean().backward()
    plain_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in plain.parameters() if p.grad is not None))
    stable_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in stable.parameters() if p.grad is not None))
    assert stable_norm < 0.25 * plain_norm
    assert stable.rule_scale.grad.norm() < plain.rule_scale.grad.norm()


def test_numeric_product_stabilization_is_disabled_in_eval_and_symbolic_modes():
    plain = _model(0.0, 0.0)
    stable = _model(3.0, 2.0)
    stable.load_state_dict(plain.state_dict())
    x = torch.randn(8, 2)

    plain.eval()
    stable.eval()
    plain.symbolic_enabled = False
    stable.symbolic_enabled = False
    assert torch.equal(plain(x), stable(x))

    plain.train()
    stable.train()
    plain.symbolic_enabled = True
    stable.symbolic_enabled = True
    assert torch.equal(plain(x), stable(x))


def test_product_jacobian_equalization_preserves_forward_and_amplifies_weak_product_gradients():
    plain = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=2, grid=4, k=2,
        numeric_factor_gradient_scale=0.0, numeric_product_gradient_scale=0.0,
        numeric_product_gradient_power=1.0, seed=7,
    )
    equalized = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=2, grid=4, k=2,
        numeric_factor_gradient_scale=0.0, numeric_product_gradient_scale=0.0,
        numeric_product_gradient_power=0.5, numeric_product_gradient_max_gain=8.0, seed=7,
    )
    equalized.load_state_dict(plain.state_dict())
    x = torch.tensor([[-0.2, 0.1], [0.3, -0.1], [0.1, 0.2]], dtype=torch.float32)
    for model in (plain, equalized):
        model.train()
        model.symbolic_enabled = False
        model.zero_grad(set_to_none=True)
    y_plain = plain(x)
    y_equalized = equalized(x)
    assert torch.equal(y_plain, y_equalized)
    y_plain.sum().backward()
    y_equalized.sum().backward()

    def edge_grad_norm(model):
        terms = [
            (p.grad.detach() ** 2).sum()
            for name, p in model.named_parameters()
            if name.startswith("numeric_edges") and p.grad is not None
        ]
        return torch.sqrt(sum(terms))

    assert edge_grad_norm(equalized) > 2.0 * edge_grad_norm(plain)
