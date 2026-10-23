import numpy as np
rng=np.random.default_rng(1)
N=1839; se1=0.95; sig2=se1**2*N  # per-episode variance of paired diff (pt^2)
R=200000
looks=np.arange(600,N+1,60)  # roughly a look every ~20 min at ~180 eps/h
for th in [0,0.5,1.0]:
    # partial sums of iid episode diffs: use gaussian increments
    inc_n=np.diff(np.concatenate([[0],looks]))
    incs=rng.normal(th*inc_n, np.sqrt(sig2*inc_n), size=(R,len(looks)))
    S=np.cumsum(incs,axis=1); m=S/looks
    single=(m[:,0]<=-3).mean(); anyl=(m<=-3).any(1).mean()
    print(f"th={th}: single look@600 {single:.3f}  repeated looks 600..1839 {anyl:.3f}")
