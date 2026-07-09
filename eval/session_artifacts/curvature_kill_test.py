"""CURVATURE KILL TEST (pre-registered).
Task: insider_trading onpolicy, episode-level honest(0) vs deceptive(1), n~171.
Trajectory: sentence/line-split of the graded output (>=3 steps; observed 8-23).
Chart: fixed 4-D deception anchors. All metrics trace-normalized to 4.

Methods (all oriented by train-fold sign, 5-fold CV over episodes):
  F0 single-point champion: full-response vector, centroid score under in-task Sw^-1
  F1 flat-I mean path energy:      mean_t ||dx||^2_I
  F2 flat-Sw^-1 mean path energy:  mean_t dx' M dx,  M = (step-level Sw)^-1
  F3 curved mean path energy:      mean_t dx' G_b(mid) dx, G_b = M + b*bump(x),
       bump(x) = sigma(1-sigma) w w' (trace-norm), w = in-task step logistic,
       b chosen on train folds from {0.25,1,4,16} (0 excluded).
  F4 position-only: mean step vector, centroid score under M.
KILL RULE (primary): curvature survives iff paired-bootstrap AUROC(F3-F2) CI > 0.
Secondary (cost-weighted): TPR at FPR<=0.2. Validity gate: report CV of the
bump magnitude across steps; if ~constant, verdict = 'no habitat' (still cut).
"""
import json, re, time, os
import numpy as np
import sentence_transformers
_O = sentence_transformers.SentenceTransformer; _C = {}
def _cst(n,*a,**k):
    if n not in _C: _C[n] = _O(n,*a,**k)
    return _C[n]
sentence_transformers.SentenceTransformer = _cst
from frontier_ops.boundary import concept_extraction as CE
from frontier_ops.boundary import semantic_extraction as SE
from earned_anchors import DEC_ANCHORS
DEC_DIMS = ["fabricated_justification","strategic_vagueness","impression_management","honest_disclosure"]
CE.CONCEPTS[:] = DEC_DIMS
SE.SEMANTIC_ANCHORS.clear(); SE.SEMANTIC_ANCHORS.update(DEC_ANCHORS)

FP = "/home/claude/deception-detection/data/rollouts/insider_trading__onpolicy__llama-70b.json"
CACHE = "/home/claude/enc_insider_steps.npz"

def split_steps(text):
    parts = [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+', text) if len(x.strip())>15]
    return parts

d = json.load(open(FP))
episodes = []
for r in d["rollouts"]:
    g = (r.get("graded_deception") or [None])[0]
    out = ((r.get("output_strs") or [""])[0] or "").strip()
    if g not in ("honest","deceptive") or not out: continue
    steps = split_steps(out)
    if len(steps) < 3: continue
    episodes.append(dict(y=0 if g=="honest" else 1, steps=steps, full=out))
Y = np.array([e["y"] for e in episodes])
print(f"episodes: {len(Y)}  honest={(Y==0).sum()}  deceptive={(Y==1).sum()}  "
      f"steps median={int(np.median([len(e['steps']) for e in episodes]))}")

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); STEP_X = list(z["step_x"]); FULL_X = z["full_x"]
else:
    enc = SE.SemanticConceptExtractor(); t0=time.time()
    STEP_X, FULL_X = [], []
    for i,e in enumerate(episodes):
        STEP_X.append(np.stack([np.array([enc.extract(s)[dd] for dd in DEC_DIMS]) for s in e["steps"]]))
        FULL_X.append(np.array([enc.extract(e["full"])[dd] for dd in DEC_DIMS]))
        if i%40==0: print(f"  encode {i}/{len(episodes)} [{time.time()-t0:.0f}s]", flush=True)
    FULL_X = np.stack(FULL_X)
    np.savez(CACHE, step_x=np.array(STEP_X,dtype=object), full_x=FULL_X)
    print(f"encoded [{time.time()-t0:.0f}s]")

def sig(z): return 1/(1+np.exp(-z))
def tnorm(M): return M*4/np.trace(M)
def auroc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,int)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    n1=y.sum(); n0=len(y)-n1
    return float((rk[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
def tpr_at_fpr(s,y,f=0.2):
    s=np.asarray(s,float); y=np.asarray(y,int)
    neg=np.sort(s[y==0])[::-1]
    k=max(int(np.floor(f*len(neg)))-1,0)
    thr=neg[k] if len(neg) else np.inf
    return float((s[y==1]>thr).mean())
def fit_logistic(X, y, ridge=1e-2, iters=60):
    Xa=np.hstack([X,np.ones((len(X),1))]); w=np.zeros(Xa.shape[1])
    for _ in range(iters):
        p=sig(Xa@w); W=p*(1-p)+1e-9
        H=(Xa.T*W)@Xa+ridge*np.eye(Xa.shape[1])
        w=w+np.linalg.solve(H,Xa.T@(y-p)-ridge*w)
    return w[:-1],w[-1]

BETAS=[0.25,1.0,4.0,16.0]
rng=np.random.default_rng(11)
perm=rng.permutation(len(Y)); folds=np.array_split(perm,5)
S={m:np.zeros(len(Y)) for m in ["F0_single","F1_pathI","F2_pathSw","F3_curved","F4_mean_pos"]}
beta_picks=[]; bump_cvs=[]
for f in range(5):
    te=folds[f]; tr=np.concatenate([folds[j] for j in range(5) if j!=f])
    # step-level pools from train episodes
    Xs_tr=np.vstack([STEP_X[i] for i in tr]); ys_tr=np.concatenate([[Y[i]]*len(STEP_X[i]) for i in tr])
    Sw=np.zeros((4,4))
    for c in (0,1):
        Z=Xs_tr[ys_tr==c]-Xs_tr[ys_tr==c].mean(0); Sw+=Z.T@Z
    M=tnorm(np.linalg.inv(Sw/max(len(ys_tr)-2,1)+1e-3*np.eye(4)))
    w,cb=fit_logistic(Xs_tr,ys_tr)
    Wout=tnorm(np.outer(w,w)+1e-9*np.eye(4))
    # episode-level covariance for F0/F4 centroids
    Xf_tr=np.stack([FULL_X[i] for i in tr]); Yf_tr=Y[tr]
    Swf=np.zeros((4,4))
    for c in (0,1):
        Z=Xf_tr[Yf_tr==c]-Xf_tr[Yf_tr==c].mean(0); Swf+=Z.T@Z
    Mf=tnorm(np.linalg.inv(Swf/max(len(tr)-2,1)+1e-3*np.eye(4)))
    mu_h=Xf_tr[Yf_tr==0].mean(0); mu_d=Xf_tr[Yf_tr==1].mean(0)
    mean_tr=np.stack([STEP_X[i].mean(0) for i in tr])
    muh_m=mean_tr[Yf_tr==0].mean(0); mud_m=mean_tr[Yf_tr==1].mean(0)

    def path_scores(idx_set, beta=None):
        out={"F1":[], "F2":[], "F3":[]}
        for i in idx_set:
            P=STEP_X[i]; D=np.diff(P,axis=0); mid=(P[1:]+P[:-1])/2
            out["F1"].append(float(np.mean(np.einsum('ni,ni->n',D,D))))
            out["F2"].append(float(np.mean(np.einsum('ni,ij,nj->n',D,M,D))))
            if beta is not None:
                pz=sig(mid@w+cb); bump=pz*(1-pz)
                e=np.einsum('ni,ij,nj->n',D,M,D)+beta*bump*np.einsum('ni,ij,nj->n',D,Wout,D)
                out["F3"].append(float(np.mean(e)))
        return out

    # beta selection on train (F3 train AUROC, orientation-aware via |AUROC-0.5|)
    tr_base=path_scores(tr)
    best=(None,-1)
    for b in BETAS:
        s3=path_scores(tr,beta=b)["F3"]
        a=abs(auroc(s3,Y[tr])-0.5)
        if a>best[1]: best=(b,a)
    beta=best[0]; beta_picks.append(beta)
    # orientation signs from train
    signs={}
    tr_s3=path_scores(tr,beta=beta)
    for nm,vals in [("F1",tr_base["F1"]),("F2",tr_base["F2"]),("F3",tr_s3["F3"])]:
        signs[nm]=1.0 if auroc(vals,Y[tr])>=0.5 else -1.0
    s0=[float((x-mu_h)@Mf@(x-mu_h)-(x-mu_d)@Mf@(x-mu_d)) for x in Xf_tr]
    signs["F0"]=1.0 if auroc(s0,Y[tr])>=0.5 else -1.0
    s4=[float((m-muh_m)@Mf@(m-muh_m)-(m-mud_m)@Mf@(m-mud_m)) for m in mean_tr]
    signs["F4"]=1.0 if auroc(s4,Y[tr])>=0.5 else -1.0
    # bump variability on train steps (validity gate)
    pz_all=sig(Xs_tr@w+cb); bmag=pz_all*(1-pz_all)
    bump_cvs.append(float(bmag.std()/max(bmag.mean(),1e-9)))
    # test scoring
    for i in te:
        P=STEP_X[i]; D=np.diff(P,axis=0); mid=(P[1:]+P[:-1])/2
        S["F1_pathI"][i]=signs["F1"]*float(np.mean(np.einsum('ni,ni->n',D,D)))
        S["F2_pathSw"][i]=signs["F2"]*float(np.mean(np.einsum('ni,ij,nj->n',D,M,D)))
        pz=sig(mid@w+cb); bump=pz*(1-pz)
        e=np.einsum('ni,ij,nj->n',D,M,D)+beta*bump*np.einsum('ni,ij,nj->n',D,Wout,D)
        S["F3_curved"][i]=signs["F3"]*float(np.mean(e))
        x=FULL_X[i]
        S["F0_single"][i]=signs["F0"]*float((x-mu_h)@Mf@(x-mu_h)-(x-mu_d)@Mf@(x-mu_d))
        m=P.mean(0)
        S["F4_mean_pos"][i]=signs["F4"]*float((m-muh_m)@Mf@(m-muh_m)-(m-mud_m)@Mf@(m-mud_m))

def paired(sa,sb,y,n=4000,seed=3):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); dd=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].min()==y[b].max(): continue
        dd.append(auroc(sa[b],y[b])-auroc(sb[b],y[b]))
    dd=np.array(dd); return dd.mean(),np.percentile(dd,[2.5,97.5])

print(f"\nbeta picks per fold: {beta_picks}")
print(f"bump magnitude CV across steps (validity gate): {np.round(bump_cvs,2)}")
print("\n== episode-level results (held-out folds) ==")
print(f"{'method':14s} {'AUROC':>7s} {'TPR@FPR0.2':>11s}")
for m,s in S.items():
    print(f"{m:14s} {auroc(s,Y):7.3f} {tpr_at_fpr(s,Y):11.3f}")
print("\n== pre-registered comparisons (paired bootstrap) ==")
for a,b,label in [("F3_curved","F2_pathSw","KILL TEST curved - flat"),
                  ("F2_pathSw","F1_pathI","metric structure on paths"),
                  ("F2_pathSw","F0_single","trajectory vs single-point"),
                  ("F4_mean_pos","F0_single","mean-position vs single-point")]:
    m,(lo,hi)=paired(S[a],S[b],Y)
    print(f"{label:28s} {m:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
np.savez("/home/claude/curvature_kill_scores.npz", Y=Y, **S)
print("\nsaved curvature_kill_scores.npz")
