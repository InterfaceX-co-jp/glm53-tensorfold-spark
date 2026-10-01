import csv,sys,re,glob,os
for d in sys.argv[1:]:
    rows=list(csv.DictReader(open(f"{d}/sb/spark_bench.csv")))
    g=lambda w,m: next((r['value'] for r in rows if r['workload']==w and r['metric']==m),None)
    dom={r['workload']:r['value'] for r in rows if r['metric']=='domain_quality'}
    log=open(f"{d}/sb.log").read()
    head=re.findall(r"=== TrueScore .*===",log); ts=re.findall(r"--- Trial Stats: .* ---",log)
    ag={}
    for r in rows:
        if r['workload'].startswith('AG-') and r['metric']=='score': ag.setdefault(r['workload'],[]).append(float(r['value']))
    print(d.split('/')[-1], '|', head[-1] if head else 'NO SCORE', '|', ts[-1] if ts else '', '| median', g('overall','median_latency'))
    print('  domains:', ' '.join(f"{k}={v}" for k,v in sorted(dom.items())))
    print('  AG:', ' '.join(f"{k}={'/'.join(f'{x:.2f}' for x in v)}" for k,v in sorted(ag.items())))
