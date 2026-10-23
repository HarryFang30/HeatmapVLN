import pickle, sys, collections, math
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from takeover import dev
H = set(sys.argv[1].split(','))
for name, f, excl in [('envdrop', 'envdrop.pkl', set()), ('hm3d', 'hm3d_base.pkl', H), ('r2r', 'round1.pkl', {'17DRP5sb8fy','5LpN3gDmAk7','gZ6f7yhEvPG','kEZ7cmS4wCh','r47D5H71a5s','VzqfbhrpDEA'})]:
    E = pickle.load(open(f, 'rb'))
    cb = collections.Counter(); db = collections.Counter(); n = 0; ntr = 0
    first_call_dev = 0
    for e in E.values():
        ep = e['ep']
        if ep['scene_id'] in excl: continue
        n += 1
        ref = ep['reference_path']
        for s in e['steps']:
            if s.get('position') is None: continue
            if dev(s['position'], ref) > 2.0:
                ntr += 1
                ci = s.get('call_index') or 0
                b = '0-2' if ci <= 2 else '3-5' if ci <= 5 else '6-11' if ci <= 11 else '12-23' if ci <= 23 else '24-47' if ci <= 47 else '48+'
                cb[b] += 1
                d = s['distance_to_goal'] or 0
                db['<3' if d < 3 else '3-6' if d < 6 else '6-10' if d < 10 else '10+'] += 1
                break
    print(name, 'n', n, 'dev2 trig', ntr, 'call bins %', {k: round(100 * v / ntr, 1) for k, v in sorted(cb.items(), key=lambda x: ['0-2','3-5','6-11','12-23','24-47','48+'].index(x[0]))}, 'dgoal bins %', {k: round(100 * v / ntr, 1) for k, v in db.items()})
