import pickle, sys, json, collections
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from dose import run
H = sys.argv[3]
for name, f, excl in [('envdrop', 'envdrop.pkl', ''), ('hm3d', 'hm3d_base.pkl', H), ('r2r', 'round1.pkl', '17DRP5sb8fy,5LpN3gDmAk7,gZ6f7yhEvPG,kEZ7cmS4wCh,r47D5H71a5s,VzqfbhrpDEA')]:
    E = pickle.load(open(f, 'rb'))
    ex = set(excl.split(',')) if excl else set()
    for c in (3.7, 4.2, 7.4, 9.9):
        o = run(E, ex, X=2.0, K=10, c=c)
        print(name, 'c', c, 'eps/h takeover', o['eps_per_slot_hour_takeover'], 'full', o['eps_per_slot_hour_full'])
