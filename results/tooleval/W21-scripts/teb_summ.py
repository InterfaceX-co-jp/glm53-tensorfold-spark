import json,sys,statistics
for p in sys.argv[1:]:
    d=json.load(open(p)); s=d['scores']
    cats=' '.join(f"{c['category']}{c['earned']}/{c['max']}" for c in s['category_scores'])
    dur=[r['duration_seconds'] for r in s['scenario_results']]
    print(f"{p}: score {s['final_score']} ({s['total_points']}/{s['max_points']}) depl {s['deployability']} resp {s['responsiveness']} median_turn {s['median_turn_ms']/1000:.2f}s tokens {s['total_tokens']} wall {sum(dur)/60:.1f}min | {cats}")
    for r in s['scenario_results']:
        if r['scenario_id'] in ('TC-07','TC-08','TC-09','TC-61'): print(f"   {r['scenario_id']} {r['status']} {r['points']} turns={r['turn_count']} {r['summary'][:90]}")
