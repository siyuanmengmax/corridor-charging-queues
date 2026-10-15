"""
Summary tables for the main factorial (E1-E3): run from the repository root.

  python corridor/experiments/summarize_main.py
"""
import json,glob,sys,numpy as np
from scipy import stats
f=sorted(glob.glob('corridor/results/main_factorial_*.json'))[-1]; R=json.load(open(f)); print(f,len(R))
def sel(**k): return [r for r in R if all(r[a]==b for a,b in k.items())]
def ci(x): x=np.array(x,float); return x.mean(),stats.t.ppf(0.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
def paired(env,load,thr,m,ref,key):
    a={r['seed']:r[key] for r in sel(env=env,load=load,threshold=thr,method=m)}; b={r['seed']:r[key] for r in sel(env=env,load=load,threshold=thr,method=ref)}
    d=[a[s]-b[s] for s in a if s in b]; mu,h=ci(d); return mu,h
M=["Greedy","StaticCap","QA-R-MPC (current)","Backlog-Greedy g=3","Fluid-MPC H=0","Fluid-MPC H=15","Fluid-MPC H=30","Fluid-MPC H=60"]
print("\n##### E1: main comparison at threshold 0.30 (mean±CI); d = paired difference of Fluid-MPC H=30 minus that method")
for env in ["phase1","phase2_sumo"]:
  for load in ["low","moderate","high"]:
    rs=sel(env=env,load=load,threshold=0.30); print(f"\n{env} / {load}  load={np.mean([r['offered_load'] for r in rs]):.2f}  requests={np.mean([r['n_requests'] for r in rs]):.0f} unserved={sum(r['n_unserved'] for r in rs)} stranded/ep={np.mean([r['n_no_station'] for r in rs]):.1f}")
    for m in M:
        x=sel(env=env,load=load,threshold=0.30,method=m)
        s=f"  {m:<20}"+"".join("%7.1f±%-5.1f"%ci([r[k] for r in x]) for k in("mean_wait_min","p95_wait_min","max_wait_min","access_mean_min"))
        if m!="Fluid-MPC H=30": s+="   dMean %+6.1f±%-4.1f dP95 %+6.1f±%-4.1f"%(paired(env,load,0.30,"Fluid-MPC H=30",m,"mean_wait_min")+paired(env,load,0.30,"Fluid-MPC H=30",m,"p95_wait_min"))
        print(s)
print("\n##### E2: mean wait by request threshold")
for env in ["phase1","phase2_sumo"]:
  for load in ["low","moderate","high"]:
    print(f"\n{env} / {load}        thr:   0.20         0.25         0.30         0.35         0.40")
    for m in ["Greedy","StaticCap","QA-R-MPC (current)","Backlog-Greedy g=3","Fluid-MPC H=30"]:
        print(f"  {m:<20}"+"".join("%6.1f±%-5.1f"%ci([r['mean_wait_min'] for r in sel(env=env,load=load,threshold=t,method=m)]) for t in [0.2,0.25,0.3,0.35,0.4]))
    print(f"  {'stranded/ep (Fluid)':<20}"+"".join("%6.1f      "%np.mean([r['n_no_station'] for r in sel(env=env,load=load,threshold=t,method='Fluid-MPC H=30')]) for t in [0.2,0.25,0.3,0.35,0.4]))
    print(f"  {'offered load':<20}"+"".join("%6.2f      "%np.mean([r['offered_load'] for r in sel(env=env,load=load,threshold=t,method='Fluid-MPC H=30')]) for t in [0.2,0.25,0.3,0.35,0.4]))
print("\n##### E3: look-ahead; paired difference vs Backlog-Greedy g=3: mean wait | p95")
for env in ["phase1","phase2_sumo"]:
  for load in ["moderate","high"]:
    print(f"\n{env} / {load}        thr:        0.20                    0.25                    0.30                    0.35                    0.40")
    for m in ["Fluid-MPC H=0","Fluid-MPC H=15","Fluid-MPC H=30","Fluid-MPC H=60"]:
        print(f"  {m:<16}"+"".join("  %+5.1f±%-4.1f|%+6.1f±%-5.1f"%(paired(env,load,t,m,"Backlog-Greedy g=3","mean_wait_min")+paired(env,load,t,m,"Backlog-Greedy g=3","p95_wait_min")) for t in [0.2,0.25,0.3,0.35,0.4]))
