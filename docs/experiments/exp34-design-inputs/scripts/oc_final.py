import random, math
rng=random.Random(7)
N=1839; S1=0.95; R=200000
def sim(th,rho,kmax):
    sd=S1*math.sqrt(1-rho); se=sd/math.sqrt(kmax); s600=S1*math.sqrt(N/600)
    c=dict(sup=0,sup1=0,neg=0,nd=0,fut=0,harm=0); ek=0
    for _ in range(R):
        if th+rng.gauss(0,s600)<=-3.0: c['harm']+=1; ek+=600/N; continue
        d=[th+rng.gauss(0,sd) for _ in range(kmax)]
        if (d[0]+d[1])/2<=0: c['fut']+=1; ek+=2; continue
        ek+=kmax; m=sum(d)/kmax
        if m-1.96*se>0:
            c['sup']+=1
            if m>=1.0: c['sup1']+=1
        elif m<=0: c['neg']+=1
        else: c['nd']+=1
    return {k:v/R for k,v in c.items()}|{'Ek':ek/R}
for kmax in (6,4):
    print('k_final',kmax)
    for rho in (0,0.25,0.5):
        for th in (0,0.5,1.0,1.5):
            o=sim(th,rho,kmax)
            print(f" rho={rho} th={th:+.1f} sup={o['sup']:.3f} sup1={o['sup1']:.3f} neg={o['neg']:.3f} nd={o['nd']:.3f} fut2={o['fut']:.3f} harm600={o['harm']:.3f} E[k]={o['Ek']:.2f}")
