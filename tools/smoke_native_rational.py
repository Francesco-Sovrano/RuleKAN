import time
import torch
from symbolic_kan.sum_product_kan import SumProductKAN, SumProductTrainingStage, SumProductRegularization, fit_sum_product_kan
from symbolic_kan.rational_sum_product_kan import RationalSumProductKAN, fit_rational_sum_product_kan

torch.manual_seed(7)
n=900
x = torch.empty(n,3).uniform_(-1.0,1.0)
y = (torch.sin(1.2*x[:,[0]]) + 0.6*x[:,[1]].square()) / (1.0 + x[:,[2]] + x[:,[2]].square())
tr,va=600,150
train_x,train_y=x[:tr],y[:tr]
val_x,val_y=x[tr:tr+va],y[tr:tr+va]
test_x,test_y=x[tr+va:],y[tr+va:]
stage = SumProductTrainingStage(
    name='smoke-numeric', steps=350, lr=3e-3,
    structure_hardening_start=1.0, structure_hardening_end=1.0,
    factor_hardening_start=1.0, factor_hardening_end=1.0,
    rule_hardening_start=1.0, rule_hardening_end=1.0,
    train_structure=False, train_symbolic=False, symbolic_enabled=False,
    regularization=SumProductRegularization(), grad_clip=1.0,
)
common=dict(in_dim=3,n_rules=8,max_factors=2,grid=8,k=3,symbolic_library=('x','x^2','sin'),seed=11)
plain=SumProductKAN(**common)
t0=time.perf_counter(); fit_sum_product_kan(plain,train_x,train_y,val_x,val_y,stages=[stage],log_every=350,show_progress=False); pt=time.perf_counter()-t0
rat=RationalSumProductKAN(**common,denominator_n_rules=8)
t0=time.perf_counter(); fit_rational_sum_product_kan(rat,train_x,train_y,val_x,val_y,stages=[stage],log_every=350,verbose=False); rt=time.perf_counter()-t0
with torch.no_grad():
    prmse=torch.sqrt(torch.mean((plain(test_x)-test_y)**2)).item()
    rpred,details=rat(test_x,return_details=True)
    rrmse=torch.sqrt(torch.mean((rpred-test_y)**2)).item()
print({'plain_rmse':prmse,'rational_rmse':rrmse,'plain_s':pt,'rational_s':rt,'den_min':details['denominator'].min().item(),'den_max':details['denominator'].max().item()})
