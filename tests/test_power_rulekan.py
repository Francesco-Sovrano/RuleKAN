import math
import torch
import torch.nn as nn

from rulekan.power_rulekan import PowerRuleKAN, inverse_power_target, safe_integer_power
from benchmarks.models import _linearized_ratio_pilot, _select_power_atoms, _merge_power_atoms, _learned_symbolic_rules


class _ExprBase(nn.Module):
    def __init__(self, fn, formula):
        super().__init__(); self.fn=fn; self.formula=formula
    def forward(self,x): return self.fn(x)
    def symbolic_formula(self, variable_names=None, **kwargs):
        import sympy as sp
        names=variable_names or [f"x{i}" for i in range(2)]
        xs=sp.symbols(" ".join(names)); xs=(xs,) if len(names)==1 else xs
        return self.formula(xs)


class _PilotNumeric(nn.Module):
    def __init__(self): super().__init__(); self.n_rules=2
    def forward(self,x,return_details=False):
        c=torch.zeros((x.shape[0],2),device=x.device,dtype=x.dtype)
        y=torch.zeros((x.shape[0],1),device=x.device,dtype=x.dtype)
        if return_details: return y,{"contributions":c}
        return y
    def hard_structure(self):
        return [
            {"factors":[{"identity":False,"variable_index":0}]},
            {"factors":[{"identity":False,"variable_index":1}]},
        ]


def test_inverse_power_target_real_domain_rules():
    y=torch.tensor([[1.0],[4.0],[9.0]])
    assert torch.allclose(inverse_power_target(y,2),torch.tensor([[1.0],[2.0],[3.0]]))
    assert torch.allclose(inverse_power_target(y,-1),1.0/y)
    assert inverse_power_target(torch.tensor([[-1.0],[1.0]]),2) is None
    assert inverse_power_target(torch.tensor([[0.0],[1.0]]),-1) is None


def test_power_rulekan_supports_multiple_outer_powered_rules():
    b0=_ExprBase(lambda x:x[:,0:1],lambda xs:xs[0])
    b1=_ExprBase(lambda x:1.0+x[:,1:2],lambda xs:1+xs[1])
    m=PowerRuleKAN([b0,b1],[[(0,2)],[(1,-1)]],scales=[2.0,3.0],bias=0.5)
    x=torch.tensor([[2.0,1.0],[3.0,3.0]])
    expected=0.5+2*x[:,0:1].square()+3/(1+x[:,1:2])
    assert torch.allclose(m(x),expected,atol=1e-6)
    assert m.chosen_powers()==[[2],[-1]]


def test_linearized_ratio_pilot_can_recover_cross_support_from_monomial_bank():
    torch.manual_seed(5)
    x=2*torch.rand(600,2)-1
    y=1.0/(1.0+0.8*x[:,0:1]*x[:,1:2])
    pilot=_linearized_ratio_pilot(
        _PilotNumeric(),x,y,structure_bank=[(0,),(1,),(0,1)],top_supports=3,ridge=1e-6,
    )
    assert [0,1] in pilot["denominator_supports"]
    assert pilot["denominator_margin"]>0.0


def test_outer_omp_can_keep_two_distinct_power_atoms():
    x=torch.linspace(-0.8,0.8,240)[:,None]
    x2=torch.cat([x,x],dim=1)
    y=(x.square()+1.0/(1.0+x)).reshape(-1,1)
    a=PowerRuleKAN([_ExprBase(lambda z:z[:,0:1],lambda xs:xs[0])],[[(0,2)]],scales=[1.0],bias=0.0)
    b=PowerRuleKAN([_ExprBase(lambda z:1.0+z[:,0:1],lambda xs:1+xs[0])],[[(0,-1)]],scales=[1.0],bias=0.0)
    candidates=[{"model":a,"val_rmse":1.0},{"model":b,"val_rmse":1.0}]
    sel,beta,vr=_select_power_atoms(candidates,x2[:160],y[:160],x2[160:],y[160:],max_outer_rules=2)
    assert len(sel)==2
    merged=_merge_power_atoms(candidates,sel,beta,reciprocal_epsilon=1e-8)
    with torch.no_grad(): rmse=torch.sqrt(torch.mean((merged(x2[160:])-y[160:])**2)).item()
    assert rmse<1e-4
    assert len(merged.terms)==2


def test_reciprocal_domain_selection_never_uses_test_coordinates():
    """A test-only singularity is an evaluation failure, not a selection input."""
    import inspect
    import benchmarks.models as bm

    base = _ExprBase(lambda x: 1.0 + x[:, 0:1], lambda xs: 1 + xs[0])
    model = PowerRuleKAN([base], [[(0, -1)]], scales=[1.0], bias=0.0)
    train_x = torch.tensor([[0.0], [0.25]])
    val_x = torch.tensor([[0.10], [0.50]])
    test_x = torch.tensor([[-1.0]])  # denominator is exactly zero only on test

    selection_margin = bm._power_reciprocal_selection_margin(model, train_x, val_x)
    test_margin = model.reciprocal_domain_margin(test_x)
    assert selection_margin >= 1.0
    assert test_margin == 0.0

    # The ratio-candidate constructor has no test-coordinate argument, and the
    # PowerRuleKAN trainer may use qx only after outer candidate selection.
    assert "qx" not in inspect.signature(bm._fit_ratio_candidate).parameters
    src = inspect.getsource(bm._train_power_rulekan)
    select_pos = src.index("_select_power_atoms")
    assert "data.test_x" not in src[:select_pos]
    assert "data.test_y" not in src[:select_pos]
    assert "reciprocal_domain_margin(qx)" not in src[:select_pos]
    assert "torch.cat([tx,vx,qx]" not in src.replace(" ", "")


def test_power_rulekan_accepts_composed_symbolic_base():
    from rulekan.composition_rulekan import Depth2CompositionAtom, ComposedRuleKAN
    from rulekan.power_rulekan import harden_symbolic_base

    atom = Depth2CompositionAtom("sin", [[(0, "x")]])
    comp = ComposedRuleKAN(atom)
    harden_symbolic_base(comp)
    model = PowerRuleKAN([comp], [[(0, 2)]], scales=[1.0], bias=0.0)
    x = torch.linspace(-0.5, 0.5, 11)[:, None]
    with torch.no_grad():
        expected = comp(x).square()
        got = model(x)
    assert torch.allclose(got, expected, atol=1e-7)
    assert "sin" in str(model.symbolic_formula(variable_names=["x0"]))


def test_fuzzy_structural_scoring_ignores_optimizer_scale_power_terms():
    base = _ExprBase(lambda x: x[:, 0:1], lambda xs: xs[0])
    tiny = PowerRuleKAN([base], [[(0, 2)]], scales=[1e-8], bias=0.0)
    visible = PowerRuleKAN([base], [[(0, 2)]], scales=[1e-3], bias=0.0)
    assert _learned_symbolic_rules(tiny) == []
    rules = _learned_symbolic_rules(visible)
    assert len(rules) == 1
    assert rules[0][0]["operator"] == "__powered_outer_term__"
