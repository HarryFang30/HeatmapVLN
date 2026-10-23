import random, math
rng=random.Random(11)
N=1839; S1=0.95; R=200000
def p_sup(kmax, rho, mu, sig, cond_min=None):
    sd=S1*math.sqrt(1-rho); se=sd/math.sqrt(kmax); s600=S1*math.sqrt(N/600)
    sup=sup1=n=0
    for _ in range(R):
        th=rng.gauss(mu,sig)
        if cond_min is not None and th<cond_min: continue
        n+=1
        if th+rng.gauss(0,s600)<=-3.0: continue
        d=[th+rng.gauss(0,sd) for _ in range(kmax)]
        if (d[0]+d[1])/2<=0: continue
        m=sum(d)/kmax
        if m-1.96*se>0:
            sup+=1
            if m>=1.0: sup1+=1
    return sup/n, sup1/n
for k in (6,4):
    for rho in (0,0.25,0.5):
        a=p_sup(k,rho,0.8,1.56)
        print(f"k={k} rho={rho} prior N(0.8,1.56): P(sup)={a[0]:.3f} P(sup>=1)={a[1]:.3f}")
