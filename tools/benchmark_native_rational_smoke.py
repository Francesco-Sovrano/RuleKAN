import json, time
import torch
from symbolic_kan.sum_product_kan import SumProductKAN, SumProductTrainingStage, SumProductRegularization, fit_sum_product_kan
from symbolic_kan.rational_sum_product_kan import RationalSumProductKAN, fit_rational_sum_product_kan


def task_quadratic_den(x):
    return (torch.sin(1.2*x[:,[0]]) + 0.6*x[:,[1]].square()) / (1.0 + x[:,[2]] + x[:,[2]].square())

def task_additive_den(x):
    return (0.7*x[:,[0]] - 0.3*x[:,[1]]) / (2.5 + 0.5*x[:,[1]] + 0.5*x[:,[2]])

stage = SumProductTrainingStage(
    name='smoke-numeric', steps=300, lr=3e-3,
    structure_hardening_start=1.0, structure_hardening_end=1.0,
    factor_hardening_start=1.0, factor_hardening_end=1.0,
    rule_hardening_start=1.0, rule_hardening_end=1.0,
    train_structure=False, train_symbolic=False, symbolic_enabled=False,
    regularization=SumProductRegularization(), grad_clip=1.0,
)
rows=[]
for task_name, task in [('quadratic_den',task_quadratic_den),('additive_den',task_additive_den)]:
  for seed in range(3):
    torch.manual_seed(1000+seed)
    x=torch.empty(900,3).uniform_(-1,1); y=task(x)
    tx,ty=x[:600],y[:600]; vx,vy=x[600:750],y[600:750]; sx,sy=x[750:],y[750:]
    common=dict(in_dim=3,n_rules=8,max_factors=2,grid=8,k=3,symbolic_library=('x','x^2','sin'),seed=seed+20)
    plain=SumProductKAN(**common)
    t=time.perf_counter(); fit_sum_product_kan(plain,tx,ty,vx,vy,stages=[stage],log_every=1000,show_progress=False); plain_s=time.perf_counter()-t
    rat=RationalSumProductKAN(**common,denominator_n_rules=8)
    t=time.perf_counter(); fit_rational_sum_product_kan(rat,tx,ty,vx,vy,stages=[stage],log_every=1000,verbose=False); rat_s=time.perf_counter()-t
    with torch.no_grad():
      p=torch.sqrt(torch.mean((plain(sx)-sy)**2)).item(); rp,det=rat(sx,return_details=True); r=torch.sqrt(torch.mean((rp-sy)**2)).item()
    rows.append(dict(task=task_name,seed=seed,plain_rmse=p,rational_rmse=r,plain_s=plain_s,rational_s=rat_s,den_min=det['denominator'].min().item(),den_max=det['denominator'].max().item()))
print(json.dumps(rows,indent=2))
for task_name in ['quadratic_den','additive_den']:
  rr=[z for z in rows if z['task']==task_name]
  print(task_name, {'plain_mean':sum(z['plain_rmse'] for z in rr)/len(rr),'rational_mean':sum(z['rational_rmse'] for z in rr)/len(rr),'time_ratio':sum(z['rational_s'] for z in rr)/sum(z['plain_s'] for z in rr)})
