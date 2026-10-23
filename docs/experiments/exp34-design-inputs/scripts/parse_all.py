import gzip, json, pickle, sys
src, out = sys.argv[1], sys.argv[2]
txt = gzip.open(src, 'rt').read()
E = {}
for b in txt.split('@@FILE ')[1:]:
    head, rest = b.split('@@CALLS', 1)
    calls, steps = rest.split('@@STEPS', 1)
    path, ej = head.split('\n', 1)
    ep = json.loads(ej)
    cl = [json.loads(l) for l in calls.strip().split('\n') if l.strip()]
    st = [json.loads(l) for l in steps.strip().split('\n') if l.strip()]
    cl = [{k: c.get(k) for k in ('call_index', 'step_id', 'kind', 'llm_output', 'distance_to_goal', 'position', 'actions', 'pixel_goal', 'native_lookdown_turns', 'terminal')} for c in cl]
    st = [{k: s.get(k) for k in ('step_id', 'action', 'call_index', 'distance_to_goal', 'position', 'phase')} for s in st]
    E[path.strip()] = dict(ep={k: ep.get(k) for k in ('scene_id', 'episode_id', 'geodesic_distance', 'instruction', 'outcome', 'protocol_seed', 'reference_path', 'goals', 'num_calls', 'num_steps', 'split')}, calls=cl, steps=st)
pickle.dump(E, open(out, 'wb'))
print(len(E))
