import numpy as np
rng=np.random.default_rng(2)
N=1839; S1=0.95; R=300000
for mode in ['episode','fixed']:
  for rho in [0,0.25,0.5]:
    res=[]
    for th in [0,1.0,1.5]:
        sd=S1*np.sqrt(1-rho); e=rng.normal(0,sd,(R,4)); d=th+e
        d600=th+e[:,0]*np.sqrt(N/600)
        alive=(d600>-3)&(d[:,:2].mean(1)>0)
        se4 = S1*np.sqrt(rho+(1-rho)/4) if mode=='episode' else S1*np.sqrt((1-rho)/4)
        sup=alive&(d.mean(1)-1.96*se4>0)
        res.append(f"th={th}:{sup.mean():.2f}")
    print('k=4',mode,'rho',rho,' '.join(res))
# dev gate union approx at th (SR) with neutral secondaries
from math import erf,sqrt
Phi=lambda z:0.5*(1+erf(z/sqrt(2)))
se=np.sqrt(0.15/1778)*100
for th in [0,1.0]:
    p_sr_kill=Phi((-1.5-th)/se); p_fut=Phi((0-th)/se)
    print('dev th',th,'SR kill',round(p_sr_kill,3),'futility(incl kill)',round(p_fut,3))
