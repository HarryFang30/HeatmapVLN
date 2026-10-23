"""Dose table under a concrete takeover rule.
Trigger = first of: dev>X m from reference polyline (3D); NP: K consecutive calls without improving best d_goal by >=0.5 m;
model STOP with d_goal>3 m; step cap. Expert (shortest path to goal) drives from trigger to goal and STOPs.
Samples: E = expert segment on InternData rule = ceil(actions/4)+1, actions = d_goal_at_trigger/0.25*R (R=1.3 turns; 1.55 upper);
D_W = model calls in the last W calls before trigger (labelled at model states); A = calls (<48) of no-takeover episodes.
Time per episode (slot): model calls*c + model steps*0.18 + expert actions*0.2 (render, no RPC)."""
import pickle, sys, math, json
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from takeover import dev, q

def run(E, excl, X=2.0, K=10, W=8, R=1.3, c=3.7):
    rows = []
    for k, e in E.items():
        ep = e['ep']
        if ep['scene_id'] in excl: continue
        oc = ep['outcome']; ref = ep['reference_path']
        steps = [s for s in e['steps'] if s.get('position') is not None]
        calls = e['calls']
        trig = None; why = None
        # call-level progress for NP trigger
        best = float('inf'); since = 0; np_call = None
        for c_ in calls:
            d = c_.get('distance_to_goal')
            if d is None: continue
            if d < best - 0.5: best = d; since = 0
            else: since += 1
            if since >= K and np_call is None and d > 3.0: np_call = c_
        for s in steps:
            if dev(s['position'], ref) > X:
                trig = dict(step=s['step_id'], call=s.get('call_index') or 0, dgoal=s['distance_to_goal']); why = 'dev'; break
        if np_call is not None and (trig is None or np_call['step_id'] < trig['step']):
            trig = dict(step=np_call['step_id'], call=np_call['call_index'], dgoal=np_call['distance_to_goal']); why = 'noprog'
        succ = oc['success'] > 0
        if trig is None and not succ:
            if (oc['steps'] or 0) >= 499:
                trig = dict(step=499, call=len(calls) - 1, dgoal=oc['distance_to_goal']); why = 'cap'
            else:
                trig = dict(step=oc['steps'], call=len(calls) - 1, dgoal=oc['distance_to_goal']); why = 'stopfar'
        r = dict(scene=ep['scene_id'], succ=succ, why=why, ncalls=len(calls), nsteps=oc['steps'] or len(steps))
        if trig:
            dg = trig['dgoal'] if trig['dgoal'] is not None else ep['geodesic_distance']
            acts = dg / 0.25 * R
            r.update(call=trig['call'], step=trig['step'], dgoal=dg, E=math.ceil(acts / 4) + 1, D=min(W, trig['call'] + 1),
                     Dall=min(48, trig['call'] + 1), model_calls=trig['call'] + 1, model_steps=trig['step'], exp_acts=acts)
        else:
            r.update(A=min(48, len(calls)), model_calls=len(calls), model_steps=r['nsteps'], exp_acts=0)
        r['t'] = r['model_calls'] * c + r['model_steps'] * 0.18 + r['exp_acts'] * 0.2
        r['t_full'] = len(calls) * c + r['nsteps'] * 0.18
        rows.append(r)
    n = len(rows); T = [r for r in rows if 'E' in r]
    out = dict(n=n, takeover_rate=round(100 * len(T) / n, 1),
               why={w: round(100 * sum(1 for r in T if r['why'] == w) / n, 1) for w in ('dev', 'noprog', 'stopfar', 'cap')},
               takeover_in_success=round(100 * sum(1 for r in T if r['succ']) / max(1, sum(r['succ'] for r in rows)), 1),
               call_at_trigger=q([r['call'] for r in T]), step_at_trigger=q([r['step'] for r in T]), dgoal_at_trigger=q([r['dgoal'] for r in T]),
               E_per_takeover_ep=q([r['E'] for r in T]))
    sE = sum(r['E'] for r in T) / n; sD = sum(r['D'] for r in T) / n; sDall = sum(r['Dall'] for r in T) / n; sA = sum(r.get('A', 0) for r in rows) / n
    out['per_cohort_ep'] = dict(E=round(sE, 2), D8=round(sD, 2), Dall48=round(sDall, 2), A=round(sA, 2),
                                core_E_D8=round(sE + sD, 2), all_E_Dall_A=round(sE + sDall + sA, 2))
    tm = sum(r['t'] for r in rows) / n; tf = sum(r['t_full'] for r in rows) / n
    out['sec_per_ep_takeover'] = round(tm, 1); out['sec_per_ep_full_model'] = round(tf, 1)
    out['eps_per_slot_hour_takeover'] = round(3600 / tm, 1); out['eps_per_slot_hour_full'] = round(3600 / tf, 1)
    out['steps_per_ep_recorded'] = round(sum(r['model_steps'] + r['exp_acts'] for r in rows) / n, 1)
    return out

if __name__ == '__main__':
    E = pickle.load(open(sys.argv[1], 'rb'))
    excl = set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] else set()
    res = {}
    for X in (2.0, 3.0):
        for K in (10,):
            res[f'X{X:g}_K{K}'] = run(E, excl, X=X, K=K)
    print(json.dumps(res, indent=1))
