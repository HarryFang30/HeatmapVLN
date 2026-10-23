"""Per-cohort stats for takeover DAgger planning.
dev(t): 3D distance from agent position at step t to the reference-path polyline.
Triggers (first occurrence): DEV_X (dev > X m), STOPFAR (model STOP with d_goal > 3 m), CAP (>= 499 steps, no stop).
Expert segment estimate from trigger: forward steps = d_geo/0.25, plus turns ~ 0.3x -> actions; samples = actions/4 + 1 (final STOP)."""
import pickle, sys, math, statistics as st, collections

def seg_dist(p, a, b):
    ab = [b[i] - a[i] for i in range(3)]; ap = [p[i] - a[i] for i in range(3)]
    L = sum(x * x for x in ab)
    t = 0.0 if L == 0 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / L))
    return math.dist(p, [a[i] + t * ab[i] for i in range(3)])

def dev(p, ref):
    if len(ref) == 1: return math.dist(p, ref[0])
    return min(seg_dist(p, ref[i], ref[i + 1]) for i in range(len(ref) - 1))

def q(xs, ps=(10, 25, 50, 75, 90)):
    if not xs: return None
    a = sorted(float(x) for x in xs)
    def pct(p):
        k = (len(a) - 1) * p / 100; f = math.floor(k); c = min(f + 1, len(a) - 1)
        return a[f] + (a[c] - a[f]) * (k - f)
    return {f'p{p}': round(pct(p), 2) for p in ps} | {'mean': round(sum(a) / len(a), 2), 'n': len(xs)}

def analyse(E, label, exclude_scenes=()):
    rows = []
    for k, e in E.items():
        ep = e['ep']
        if ep['scene_id'] in exclude_scenes: continue
        oc = ep['outcome'] or {}
        ref = ep['reference_path']
        steps = [s for s in e['steps'] if s.get('position') is not None]
        calls = e['calls']
        devs = [dev(s['position'], ref) for s in steps]
        r = dict(key=k, scene=ep['scene_id'], succ=oc.get('success', 0) > 0, os=oc.get('oracle_success', 0) > 0,
                 nsteps=oc.get('steps'), ncalls=len(calls), geo=ep['geodesic_distance'],
                 dgoal_end=oc.get('distance_to_goal'), maxdev=max(devs) if devs else 0)
        last = calls[-1] if calls else None
        stopped = any((c.get('llm_output') or '').strip().upper().startswith('STOP') for c in calls[-1:])
        r['stopped'] = stopped
        r['cap'] = (oc.get('steps') or 0) >= 499
        for X in (1.0, 2.0, 3.0):
            trig = None
            for s, d in zip(steps, devs):
                if d > X:
                    trig = s; break
            if trig is not None:
                ci = trig.get('call_index')
                r[f'dev{X:g}'] = dict(step=trig['step_id'], call=ci, dgoal=trig['distance_to_goal'])
            else:
                r[f'dev{X:g}'] = None
        # stop far
        if stopped and not r['succ'] and (r['dgoal_end'] or 0) > 3:
            r['stopfar'] = dict(step=oc.get('steps'), call=len(calls) - 1, dgoal=r['dgoal_end'])
        else:
            r['stopfar'] = None
        rows.append(r)
    n = len(rows)
    out = {'label': label, 'episodes': n, 'scenes': len({r['scene'] for r in rows})}
    out['SR'] = round(100 * sum(r['succ'] for r in rows) / n, 2)
    out['OS'] = round(100 * sum(r['os'] for r in rows) / n, 2)
    out['cap_rate'] = round(100 * sum(r['cap'] for r in rows) / n, 2)
    out['stopfar_rate'] = round(100 * sum(r['stopfar'] is not None for r in rows) / n, 2)
    out['calls_per_ep'] = q([r['ncalls'] for r in rows])
    out['calls_per_ep_succ'] = q([r['ncalls'] for r in rows if r['succ']])
    out['calls_per_ep_fail'] = q([r['ncalls'] for r in rows if not r['succ']])
    out['steps_per_ep'] = q([r['nsteps'] for r in rows if r['nsteps'] is not None])
    out['geo'] = q([r['geo'] for r in rows])
    out['steps_per_call'] = round(sum(r['nsteps'] or 0 for r in rows) / max(1, sum(r['ncalls'] for r in rows)), 2)
    out['maxdev'] = q([r['maxdev'] for r in rows])
    for X in ('1', '2', '3'):
        key = f'dev{X}'
        trig = [r for r in rows if r[key] is not None]
        out[key] = {
            'trigger_rate': round(100 * len(trig) / n, 2),
            'trigger_rate_succ': round(100 * sum(1 for r in trig if r['succ']) / max(1, sum(r['succ'] for r in rows)), 2),
            'trigger_rate_fail': round(100 * sum(1 for r in trig if not r['succ']) / max(1, sum(not r['succ'] for r in rows)), 2),
            'fail_covered': round(100 * sum(1 for r in trig if not r['succ']) / max(1, sum(not r['succ'] for r in rows)), 2),
            'call_at_trigger': q([r[key]['call'] for r in trig if r[key]['call'] is not None]),
            'step_at_trigger': q([r[key]['step'] for r in trig]),
            'dgoal_at_trigger': q([r[key]['dgoal'] for r in trig if r[key]['dgoal'] is not None]),
        }
        # union trigger: dev X or stopfar or cap
        uni = []
        for r in rows:
            t = r[key] or r['stopfar']
            if t is None and r['cap']:
                t = dict(step=499, call=r['ncalls'] - 1, dgoal=r['dgoal_end'])
            if t is not None: uni.append((r, t))
        # samples: pre-takeover model calls (call index at trigger) + expert segment
        exp_samples = []; pre_calls = []
        for r, t in uni:
            dg = t['dgoal'] if t['dgoal'] is not None else r['geo']
            actions = (dg / 0.25) * 1.3
            exp_samples.append(actions / 4 + 1)
            pre_calls.append(t['call'] if t['call'] is not None else r['ncalls'])
        out[key]['union_trigger_rate'] = round(100 * len(uni) / n, 2)
        out[key]['union_fail_covered'] = round(100 * sum(1 for r, t in uni if not r['succ']) / max(1, sum(not r['succ'] for r in rows)), 2)
        out[key]['expert_samples_per_takeover_ep'] = q(exp_samples)
        out[key]['pre_takeover_calls_per_takeover_ep'] = q(pre_calls)
        tot_exp = sum(exp_samples); tot_pre = sum(pre_calls)
        nontake_calls = sum(r['ncalls'] for r in rows if not any(r is u for u, _ in uni))
        out[key]['samples_per_cohort_ep'] = {
            'expert_only': round(tot_exp / n, 2),
            'expert_plus_pre_takeover_calls': round((tot_exp + tot_pre) / n, 2),
            'all_calls_plus_expert': round((tot_exp + tot_pre + nontake_calls) / n, 2),
        }
    return out, rows

if __name__ == '__main__':
    import json
    E = pickle.load(open(sys.argv[1], 'rb'))
    excl = set(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
    out, rows = analyse(E, sys.argv[2], excl)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    pickle.dump(rows, open(sys.argv[1].replace('.pkl', '_rows.pkl'), 'wb'))
