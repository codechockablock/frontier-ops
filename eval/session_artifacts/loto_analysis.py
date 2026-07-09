import numpy as np
from frontier_ops.boundary import concept_extraction as CE
from frontier_ops.boundary.constitution import ConstitutionSpec, Boundary, CrossTerm, ConstitutionalMetric

DEC_DIMS = ["fabricated_justification","strategic_vagueness","impression_management","honest_disclosure"]
CE.CONCEPTS[:] = DEC_DIMS
TASKS = ["roleplaying","ai_liar","insider_trading","sandbagging"]
data = {t: (lambda z: (z["X"], z["Y"]))(np.load(f"/home/claude/enc_{t}.npz")) for t in TASKS}

def fit_logistic(X, y, ridge=1e-2, iters=60):
    Xa = np.hstack([X, np.ones((len(X),1))]); w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-(Xa@w))); W = p*(1-p)+1e-9
        H = (Xa.T*W)@Xa + ridge*np.eye(Xa.shape[1])
        w = w + np.linalg.solve(H, Xa.T@(y-p) - ridge*w)
    return w[:-1], w[-1]
def auroc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    n1=y.sum(); n0=len(y)-n1
    return float((rk[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
def sig(z): return 1/(1+np.exp(-z))
def norm_floor(M, eps=0.05):
    d=M.shape[0]; tr=np.trace(M)
    M = M*d/tr if tr>1e-12 else np.eye(d)
    return M + eps*np.eye(d)
def cdist2(Xe, mu, M): return np.einsum('ni,ij,nj->n', Xe-mu, M, Xe-mu)

risk = {}
print("per-task logistic risk models (in-sample sanity):")
for t in TASKS:
    X,Y = data[t]; w,c = fit_logistic(X,Y)
    p = sig(X@w+c)
    risk[t] = dict(w=w, c=c, fw=float(np.mean(p*(1-p))))
    wn = w/ (np.linalg.norm(w)+1e-12)
    print(f"  {t:16s} AUROC={auroc(X@w+c,Y):.3f} |w|={np.linalg.norm(w):6.2f} fw={risk[t]['fw']:.3f} "
          f"w_hat=[{', '.join(f'{v:+.2f}' for v in wn)}]")

dec_const = ConstitutionSpec(name="deception_v0",
    boundaries=[Boundary("fabricated_justification",0.45,5.0,3.0),
                Boundary("strategic_vagueness",0.50,4.0,2.0),
                Boundary("impression_management",0.50,4.0,2.0)],
    cross_terms=[CrossTerm(("fabricated_justification","impression_management"),3.0)])
CM = ConstitutionalMetric(dec_const, dim_names=DEC_DIMS)

METHODS = ["identity","asserted_G","transfer_cov","intask_cov","earned_flat","earned_curved"]
def loto_eval(hold):
    Xh, Yh = data[hold]
    train_tasks = [k for k in TASKS if k != hold]
    Ge = np.zeros((4,4))
    for k in train_tasks: Ge += risk[k]["fw"]*np.outer(risk[k]["w"],risk[k]["w"])
    Ge = norm_floor(Ge)
    Sw = np.zeros((4,4)); dof=0
    for k in train_tasks:
        Xk,Yk = data[k]
        for c in (0,1):
            Z = Xk[Yk==c]-Xk[Yk==c].mean(0); Sw += Z.T@Z; dof += max((Yk==c).sum()-1,0)
    Mtrans = np.linalg.inv(Sw/max(dof,1) + 1e-3*np.eye(4))
    ws = [(risk[k]["w"], risk[k]["c"]) for k in train_tasks]
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(Yh)); folds = np.array_split(perm,5)
    S = {m: np.zeros(len(Yh)) for m in METHODS}
    for f in range(5):
        te=folds[f]; tr=np.concatenate([folds[j] for j in range(5) if j!=f])
        Xtr,Ytr,Xte = Xh[tr],Yh[tr],Xh[te]
        mu_h=Xtr[Ytr==0].mean(0); mu_d=Xtr[Ytr==1].mean(0)
        Swt=np.zeros((4,4))
        for c in (0,1):
            Z=Xtr[Ytr==c]-Xtr[Ytr==c].mean(0); Swt+=Z.T@Z
        Mint=np.linalg.inv(Swt/max(len(tr)-2,1)+1e-3*np.eye(4))
        Gass=np.mean([CM.tensor_at(x) for x in Xtr[::max(len(tr)//150,1)]],axis=0)
        for nm,M in [("identity",np.eye(4)),("asserted_G",Gass),("transfer_cov",Mtrans),
                     ("intask_cov",Mint),("earned_flat",Ge)]:
            S[nm][te]=cdist2(Xte,mu_h,M)-cdist2(Xte,mu_d,M)
        for i in te:
            x=Xh[i]; Gx=np.zeros((4,4))
            for w,c in ws:
                p=sig(x@w+c); Gx+=p*(1-p)*np.outer(w,w)
            Gx=norm_floor(Gx)
            S["earned_curved"][i]=float((x-mu_h)@Gx@(x-mu_h)-(x-mu_d)@Gx@(x-mu_d))
    return S,Yh

def paired(sa,sb,y,n=3000,seed=1):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); d=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].min()==y[b].max(): continue
        d.append(auroc(sa[b],y[b])-auroc(sb[b],y[b]))
    d=np.array(d); return d.mean(), np.percentile(d,[2.5,97.5])

print("\n== LOTO transfer: AUROC on held-out task ==")
print(f"{'held-out':16s}"+"".join(f"{m:>14s}" for m in METHODS))
all_S={}
for hold in TASKS:
    S,Yh=loto_eval(hold); all_S[hold]=(S,Yh)
    print(f"{hold:16s}"+"".join(f"{auroc(S[m],Yh):14.3f}" for m in METHODS))

print("\n== paired diffs (held-out): earned_flat vs identity | vs asserted_G | vs transfer_cov ==")
for hold in TASKS:
    S,Yh=all_S[hold]
    a,(l1,h1)=paired(S["earned_flat"],S["identity"],Yh)
    b,(l2,h2)=paired(S["earned_flat"],S["asserted_G"],Yh)
    c,(l3,h3)=paired(S["earned_flat"],S["transfer_cov"],Yh)
    print(f"{hold:16s} {a:+.3f} [{l1:+.3f},{h1:+.3f}] | {b:+.3f} [{l2:+.3f},{h2:+.3f}] | {c:+.3f} [{l3:+.3f},{h3:+.3f}]")

Gfull=np.zeros((4,4))
for k in TASKS: Gfull+=risk[k]["fw"]*np.outer(risk[k]["w"],risk[k]["w"])
Gn=Gfull*4/np.trace(Gfull)
ev,V=np.linalg.eigh(Gn)
print("\n== earned G (all 4 tasks): eigenstructure ==")
for i in range(3,-1,-1):
    print(f"  eig={ev[i]:.3f}  ["+", ".join(f"{DEC_DIMS[j][:12]}:{V[j,i]:+.2f}" for j in range(4))+"]")
np.savez("/home/claude/earned_metric.npz", G=Gn, dims=np.array(DEC_DIMS),
         **{f"w_{k}": risk[k]["w"] for k in TASKS})
print("saved earned_metric.npz")
