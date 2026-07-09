"""External half of the live question, done properly in 384-D.
Per task: deception prototype direction = unit(mean_dec - mean_hon) in raw MiniLM space.
Outputs: (1) in-task CV AUROC of the direction, (2) 4x4 cross-task transfer AUROC
(raw projection, no orientation flip; <0.5 = anti-aligned), (3) pairwise cosine
matrix with episode-bootstrap CIs, (4) nulls: random-direction E|cos| in d=384 vs d=4.
"""
import numpy as np
TASKS = ["roleplaying","ai_liar","insider_trading","sandbagging"]
data = {t:(lambda z:(z["E"],z["Y"]))(np.load(f"/home/claude/raw_{t}.npz")) for t in TASKS}

def auroc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    n1=y.sum(); n0=len(y)-n1
    return float((rk[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
def proto(E,Y):
    d = E[Y==1].mean(0)-E[Y==0].mean(0)
    return d/(np.linalg.norm(d)+1e-12)

print("== in-task 5-fold CV AUROC of prototype direction (384-D) ==")
rng=np.random.default_rng(5)
for t in TASKS:
    E,Y=data[t]; perm=rng.permutation(len(Y)); folds=np.array_split(perm,5)
    s=np.zeros(len(Y))
    for f in range(5):
        te=folds[f]; tr=np.concatenate([folds[j] for j in range(5) if j!=f])
        w=proto(E[tr],Y[tr]); s[te]=E[te]@w
    print(f"  {t:16s} {auroc(s,Y):.3f}   (4-D chart in-task was ~0.62)")

print("\n== cross-task transfer AUROC (rows=trained on, cols=eval; raw, no flip) ==")
W={t:proto(*data[t]) for t in TASKS}
print(f"{'':16s}"+"".join(f"{t[:9]:>11s}" for t in TASKS))
for a in TASKS:
    row=[]
    for b in TASKS:
        E,Y=data[b]; row.append(auroc(E@W[a],Y))
    print(f"{a:16s}"+"".join(f"{v:11.3f}" for v in row))

print("\n== pairwise cos(w_a, w_b), 384-D, with episode-bootstrap 95% CI ==")
B=300; boot={t:[] for t in TASKS}
for t in TASKS:
    E,Y=data[t]; idx=np.arange(len(Y)); r2=np.random.default_rng(9)
    for _ in range(B):
        b=r2.choice(idx,len(idx),replace=True)
        if Y[b].min()==Y[b].max(): boot[t].append(W[t]); continue
        boot[t].append(proto(E[b],Y[b]))
print(f"{'':16s}"+"".join(f"{t[:9]:>19s}" for t in TASKS))
offdiag=[]
for i,a in enumerate(TASKS):
    cells=[]
    for j,b in enumerate(TASKS):
        c=float(W[a]@W[b])
        cs=[float(boot[a][k]@boot[b][k]) for k in range(B)]
        lo,hi=np.percentile(cs,[2.5,97.5])
        cells.append(f"{c:+.2f}[{lo:+.2f},{hi:+.2f}]")
        if j>i: offdiag.append(abs(c))
    print(f"{a:16s}"+"".join(f"{x:>19s}" for x in cells))
d=384
print(f"\nmean |cos| off-diag (384-D): {np.mean(offdiag):.3f}")
print(f"null E|cos| random dirs: d=384 -> {np.sqrt(2/(np.pi*d)):.3f} ; d=4 -> {np.sqrt(2/(np.pi*4)):.3f} (4-D chart observed 0.354)")
