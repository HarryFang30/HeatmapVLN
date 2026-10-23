import numpy as np
rng=np.random.default_rng(0)
N=1839; S1=0.95; R=400000
def sim(theta, rho, se_mode, kmax=6, early=True, fut=True, interim600=True, se1=S1):
    # theta = fixed-set true effect (mean over the 1839 episodes, averaged over seeds)
    # per-seed noise: within-episode seed noise only
    sd_seed = se1*np.sqrt(1-rho)
    e = rng.normal(0, sd_seed, size=(R,kmax))
    d = theta + e                        # per-seed paired delta
    # 600-ep interim on seed 1: extra subset noise (both seed noise and heterogeneity of subset)
    d600 = theta + rng.normal(0, se1*np.sqrt(N/600 - 1), size=R)*0 + e[:,0]*np.sqrt(N/600)  # approx
    out = {'sup':0,'ref':0,'nd':0,'fut2':0,'stop600':0}
    seeds=np.zeros(R)
    alive=np.ones(R,bool)
    if interim600:
        s=alive & (d600<=-3.0); out['stop600']=s.mean(); alive&=~s; seeds[s]=600/N
    cum=np.cumsum(d,axis=1)
    def se(k):
        if se_mode=='episode': return se1*np.sqrt(rho+(1-rho)/k)
        else: return se1*np.sqrt((1-rho)/k)
    m2=cum[:,1]/2
    if fut:
        s=alive&(m2<=0); out['fut2']=s.mean(); alive&=~s; seeds[s]=2
    m4=cum[:,3]/4
    sup=np.zeros(R,bool)
    if early:
        s=alive&(m4-2.5*se(4)>0); sup|=s; alive&=~s; seeds[s]=4
    m6=cum[:,5]/6
    s=alive&(m6-1.96*se(6)>0); sup|=s; alive_after=alive&~s
    ref=alive_after&(m6<=0); nd=alive_after&~(m6<=0)
    seeds[alive]=6
    out['sup']=sup.mean(); out['ref']=ref.mean(); out['nd']=nd.mean(); out['Eseeds']=seeds.mean()
    return out
for mode in ['episode','fixed']:
  for rho in [0,0.25,0.5]:
    print(mode,'rho',rho)
    for th in [0,0.5,1.0,1.5,2.0]:
        o=sim(th,rho,mode)
        print(f"  th={th:+.1f} sup={o['sup']:.3f} ref6={o['ref']:.3f} nd={o['nd']:.3f} fut2={o['fut2']:.3f} stop600={o['stop600']:.3f} E[k]={o['Eseeds']:.2f}")
# k=4 only variant (user item 4)
print('k=4 final, futility at 2, no early')
for mode in ['episode','fixed']:
  for rho in [0,0.25,0.5]:
    res=[]
    for th in [0,1.0,1.5]:
        sd=S1*np.sqrt(1-rho); e=rng.normal(0,sd,(R,4)); d=th+e
        d600=e[:,0]*np.sqrt(N/600)+th
        alive=~(d600<=-3); alive&=~(d.mean(1)*0+ (d[:,:2].mean(1)<=0))
        se4 = S1*np.sqrt(rho+(1-rho)/4) if mode=='episode' else S1*np.sqrt((1-rho)/4)
        sup=alive&(d.mean(1)-1.96*se4>0)
        res.append(f"th={th} sup={sup.mean():.3f}")
    print(mode,rho,res)
