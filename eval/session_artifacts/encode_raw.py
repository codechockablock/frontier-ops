import json, time, os
import numpy as np
from sentence_transformers import SentenceTransformer
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
            y = 0 if g<=2 else (1 if g>=6 else None)
            if y is None: continue
        elif g=="honest": y=0
        elif g=="deceptive": y=1
        else: continue
        T.append(out); Y.append(y)
    return T, np.array(Y)
m = SentenceTransformer("all-MiniLM-L6-v2")
for name, fp in TASKS.items():
    out=f"/home/claude/raw_{name}.npz"
    if os.path.exists(out): print(name,"cached"); continue
    T,Y = load(fp); t0=time.time()
    E = m.encode(T, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    np.savez(out, E=E, Y=Y); print(f"{name:16s} n={len(Y)} dim={E.shape[1]} [{time.time()-t0:.0f}s]", flush=True)
