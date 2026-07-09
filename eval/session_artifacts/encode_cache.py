import json, time, os
import numpy as np
import sentence_transformers
_O = sentence_transformers.SentenceTransformer; _C = {}
def _cst(n,*a,**k):
    if n not in _C: _C[n] = _O(n,*a,**k)
    return _C[n]
sentence_transformers.SentenceTransformer = _cst
from frontier_ops.boundary import concept_extraction as CE
from frontier_ops.boundary import semantic_extraction as SE

DEC_DIMS = ["fabricated_justification","strategic_vagueness","impression_management","honest_disclosure"]
DEC_ANCHORS = json.load(open("/home/claude/dec_anchors.json")) if os.path.exists("/home/claude/dec_anchors.json") else None
from earned_anchors import DEC_ANCHORS  # single source of truth
CE.CONCEPTS[:] = DEC_DIMS
SE.SEMANTIC_ANCHORS.clear(); SE.SEMANTIC_ANCHORS.update(DEC_ANCHORS)

RDIR = "/home/claude/deception-detection/data/rollouts/"
TASKS = {
 "roleplaying":  RDIR+"roleplaying__plain__llama-70b-3.3.json",
 "ai_liar":      RDIR+"ai_liar__original_without_answers__llama-70b-3.3.json",
 "insider_trading": RDIR+"insider_trading__onpolicy__llama-70b.json",
 "sandbagging":  RDIR+"sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
}
def load(fp):
    d = json.load(open(fp)); T, Y = [], []
    for r in d["rollouts"]:
        g = (r.get("graded_deception") or [None])[0]
        out = ((r.get("output_strs") or [""])[0] or "").strip()
        if not out: continue
        if isinstance(g,(int,float)):
            if g<=2: y=0
            elif g>=6: y=1
            else: continue
        elif g=="honest": y=0
        elif g=="deceptive": y=1
        else: continue
        T.append(out); Y.append(y)
    return T, np.array(Y)

enc = SE.SemanticConceptExtractor()
for name, fp in TASKS.items():
    out = f"/home/claude/enc_{name}.npz"
    if os.path.exists(out):
        print(name, "cached"); continue
    T, Y = load(fp)
    t0=time.time(); X=[]
    for i,t in enumerate(T):
        s = enc.extract(t)           # ONE encode per text
        X.append([s[d] for d in DEC_DIMS])
        if i % 200 == 0: print(f"  {name} {i}/{len(T)} [{time.time()-t0:.0f}s]", flush=True)
    np.savez(out, X=np.array(X), Y=Y)
    print(f"{name:16s} n={len(Y)} saved [{time.time()-t0:.0f}s]", flush=True)
