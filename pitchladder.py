# Minimal next‑pitch TYPE model (hierarchical counts + per‑pitcher sequence prior)

# What this file does:
# - Reads raw pitch-by-pitch CSV(s)
# - Builds only "pre-pitch" features 
# - Trains a tiny, explainable model (Bayes Pitch Ladder)
# - Evaluates on a 70/30 split
# - Predicts probabilities for one situation

import argparse, pickle, json
from pathlib import Path
import numpy as np, pandas as pd

# ---- Canonical classes & mapping ----
# Why: Pitch labels vary (FA/F4/FF etc). Canonicalize to 7 families for interpretability.
PITCH_CANON = ["FF","CT","SI","SL","CU","CH","KN"]
PITCH_MAP = {"FF":"FF","FA":"FF","F4":"FF","4S":"FF","FC":"CT","CT":"CT","CUT":"CT",
             "FT":"SI","SI":"SI","SINK":"SI","2S":"SI","SL":"SL","ST":"SL","SW":"SL",
             "CU":"CU","KC":"CU","CB":"CU","CH":"CH","FS":"CH","SPL":"CH","SP":"CH",
             "KN":"KN","KB":"KN"}
# Normalize any raw pitch label to one of the 7 canonical classes.
def canon(x): 
    if pd.isna(x): return None
    return PITCH_MAP.get(str(x).strip().upper())

# ---- IO ----
# Read a single CSV or a folder of CSVs into one DataFrame, and attach a 'date' column.
def read_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    if p.is_file(): df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    else:
        parts = []
        for f in sorted(p.rglob("*.csv")):
            try: parts.append(pd.read_csv(f, low_memory=False, on_bad_lines="skip"))
            except Exception: pass
        if not parts: raise RuntimeError("No readable CSVs in folder.")
        df = pd.concat(parts, ignore_index=True)
    if "date" in df: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "start_tfs" in df: df["date"] = pd.to_datetime(df["start_tfs"], errors="coerce")
    else: df["date"] = pd.to_datetime("2011-01-01")
    return df

# ---- Features (pre‑pitch only) ----
# Create only pre‑pitch features (no outcomes) for next‑pitch prediction.
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pitch_type" not in df: raise ValueError("Need 'pitch_type'.")
    df["pitch_type_canon"] = df["pitch_type"].map(canon)
    # Sort so "previous pitch" is well-defined
    sk = [c for c in ["date","game_pk","at_bat_num","pitch_id"] if c in df]
    df = df.sort_values(sk or ["date"]).reset_index(drop=True)
    # Choose grouping scope for the lag. Prefer per-at-bat, else fall back.
    if {"game_pk","at_bat_num"}.issubset(df): grp = ["game_pk","at_bat_num"]
    elif {"game_pk","pitcher_id"}.issubset(df): grp = ["game_pk","pitcher_id"]
    elif "game_pk" in df: grp = ["game_pk"]
    else: grp = None
    prev = (lambda c: df.groupby(grp)[c].shift(1)) if grp else (lambda c: df[c].shift(1))
    # Short memory (one-step sequence) for next-pitch
    df["prev_pitch_type"] = prev("pitch_type").map(canon).fillna("NONE")
    # Handedness defaults (avoid missing)
    if "stand" in df: df["stand"] = df["stand"].fillna("R").astype(str)
    else: df["stand"] = "R"
    if "p_throws" in df: df["p_throws"] = df["p_throws"].fillna("R").astype(str)
    else: df["p_throws"] = "R"
    # Numeric pre-pitch context (coerce + fill for robustness)
    for c in ["balls","strikes","inning","outs","pcount_at_bat","pcount_pitcher"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c in df else 0
    df["top"] = pd.to_numeric(df.get("top",0), errors="coerce").fillna(0).astype(int)
    # Base state flags (0/1) for runners on base
    for b in ["on_1b","on_2b","on_3b"]:
        df[b] = (df[b].notna().astype(int) if b in df else 0)
    # Times-through-order (TTO) (1, 2, 3+)
    if {"game_pk","pitcher_id","at_bat_num"}.issubset(df):
        tto = df.groupby(["game_pk","pitcher_id"])["at_bat_num"].rank("dense", ascending=True)
        df["tto_bucket"] = pd.cut(tto, [0,9,18,1e9], labels=["1","2","3+"]).astype(str)
    else: df["tto_bucket"] = "1"
    return df

# ---- Model ----
class BayesLadder:
    def __init__(self, m=120.0, lam=0.30):
        self.m, self.lam = float(m), float(lam)
        self.classes, self.K = PITCH_CANON, len(PITCH_CANON)
        self.levels = [[],
                       ["pitcher_id"],
                       ["pitcher_id","stand","p_throws"],
                       ["pitcher_id","balls","strikes"],
                       ["pitcher_id","balls","strikes","prev_pitch_type"],
                       ["pitcher_id","balls","strikes","prev_pitch_type","on_1b","on_2b","on_3b"]]
        self.tables, self.league, self.trans, self.leagueT = [], None, {}, None

    # Fit the model by counting events at each ladder level and building per-pitcher transition matrices for the sequence prior.
    def fit(self, df: pd.DataFrame):
        df = df[~df["pitch_type_canon"].isna()].copy()
        self.league = df["pitch_type_canon"].value_counts().reindex(self.classes, fill_value=0).to_numpy(float)
        self.tables = []
        for cols in self.levels:
            if not cols: self.tables.append({(): self.league}); continue
            g = (df.groupby(cols)["pitch_type_canon"].value_counts().unstack(fill_value=0)
                   .reindex(columns=self.classes, fill_value=0))
            tbl = { (ix if isinstance(ix,tuple) else (ix,)): g.loc[ix].to_numpy(float) for ix in g.index }
            self.tables.append(tbl)
        # per‑pitcher transitions
        idx = {c:i for i,c in enumerate(self.classes)}
        self.trans = {}
        for pid, grp in df.groupby("pitcher_id"):
            T = np.zeros((self.K,self.K))
            yp = grp["prev_pitch_type"].map(idx).fillna(-1).to_numpy()
            y  = grp["pitch_type_canon"].map(idx).to_numpy()
            for a,b in zip(yp,y):
                if a>=0 and b>=0: T[int(a),int(b)] += 1
            self.trans[pid] = T
        if len(self.trans):
            self.leagueT = np.sum(np.stack(list(self.trans.values())), axis=0)
        else:
            self.leagueT = np.ones((self.K,self.K))
        return self
    
    # Apply empirical-Bayes smoothing down the ladder.
    def _smooth(self, contexts):
        p = contexts[0][0].astype(float); p = p/p.sum() if p.sum()>0 else np.ones(self.K)/self.K
        for cnts,n in contexts[1:]:
            n = float(n); emp = (cnts/max(n,1.0)) if n>0 else np.zeros(self.K)
            p = (n*emp + self.m*p) / (n + self.m)
        return p/p.sum()

    # Gather the sequence of (counts, n) for this row's contexts at each ladder level.
    def _lookup(self, row):
        ctxs=[]
        for cols,tbl in zip(self.levels,self.tables):
            key = tuple(row.get(c,None) for c in cols) if cols else ()
            counts = tbl.get(key, np.zeros(self.K))
            ctxs.append((counts, counts.sum()))
        return ctxs
    
    # Build a (shrunk) per-pitcher sequence prior: P(next | prev, pitcher).
    def _seq(self, pid, prev_t):
        if (pid not in self.trans) or (prev_t not in self.classes): return np.ones(self.K)/self.K
        i = self.classes.index(prev_t); row = self.trans[pid][i].copy()
        league_row = self.leagueT.sum(axis=0); league_row /= max(league_row.sum(),1.0)
        p = row + 10.0*league_row
        return p/p.sum() if p.sum()>0 else np.ones(self.K)/self.K

    # Predict a 7-way probability distribution for each pre-pitch row in X.
    def predict_proba(self, X: pd.DataFrame):
        rows = X.to_dict("records"); P=[]
        for r in rows:
            p_sit = self._smooth(self._lookup(r))
            prev_t, pid = r.get("prev_pitch_type","NONE"), r.get("pitcher_id",None)
            if pid is not None and prev_t in self.classes:
                P.append(((1-self.lam)*p_sit + self.lam*self._seq(pid,prev_t)))
            else: P.append(p_sit)
        return np.vstack(P)

# ---- Eval & CLI ----
# Chronological 70/30 split (future pitches unseen).
def split_time(df: pd.DataFrame):
    cut = df["date"].quantile(0.70) if df["date"].notna().any() else None
    tr = df[df["date"]<=cut] if cut is not None else df.iloc[:int(0.7*len(df))]
    te = df[df["date"]> cut] if cut is not None else df.iloc[int(0.7*len(df)):]
    return tr, te

# compute evals
def evaluate(y, P):
    eps=1e-12; P=np.clip(P,eps,1.0)
    ll = float(-(np.log(P[np.arange(len(y)), y]).mean()))
    top1 = float((y==P.argmax(1)).mean())
    top2 = float(np.mean([yt in np.argsort(-p)[:2] for yt,p in zip(y,P)]))
    return {"logloss":ll,"top1":top1,"top2":top2}

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_tr = sub.add_parser("train");  p_tr.add_argument("--data", required=True); p_tr.add_argument("--out", default="./pitch_models")
    p_ev = sub.add_parser("eval");   p_ev.add_argument("--data", required=True); p_ev.add_argument("--model", required=True)
    p_pr = sub.add_parser("predict");p_pr.add_argument("--model", required=True)
    for k,v,t in [("pitcher_id",None,str),("stand","R",str),("p_throws","R",str),
                  ("balls",0,int),("strikes",0,int),("prev_pitch_type","NONE",str),
                  ("on_1b",0,int),("on_2b",0,int),("on_3b",0,int),("tto_bucket","1",str)]:
        p_pr.add_argument(f"--{k}", type=t, default=v)
    args = ap.parse_args()

    if args.cmd=="train":
        df = build_features(read_data(args.data))
        data = df[~df["pitch_type_canon"].isna()].copy()
        mdl = BayesLadder(m=120.0, lam=0.30).fit(data)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        league = (data["pitch_type_canon"].value_counts()
                  .reindex(PITCH_CANON, fill_value=0).to_numpy(float))
        league = league/league.sum() if league.sum()>0 else np.ones(len(PITCH_CANON))/len(PITCH_CANON)
        pickle.dump({"model":mdl,"league":league}, open(out/"ladder.pkl","wb"))
        tr, te = split_time(data)
        use_cols = sorted(set(c for cols in mdl.levels for c in cols))
        Xte = te[use_cols].copy()
        yte = pd.Categorical(te["pitch_type_canon"], categories=PITCH_CANON).codes
        met = evaluate(yte, mdl.predict_proba(Xte))
        print(f"[LADDER] top1={met['top1']:.3f}  top2={met['top2']:.3f}  logloss={met['logloss']:.3f}")
        json.dump(met, open(out/"ladder_metrics.json","w"), indent=2)

    elif args.cmd=="eval":
        pack = pickle.load(open(args.model,"rb")); mdl = pack["model"]
        df = build_features(read_data(args.data)); data = df[~df["pitch_type_canon"].isna()].copy()
        tr, te = split_time(data)
        use_cols = sorted(set(c for cols in mdl.levels for c in cols))
        Xte = te[use_cols].copy()
        yte = pd.Categorical(te["pitch_type_canon"], categories=PITCH_CANON).codes
        met = evaluate(yte, mdl.predict_proba(Xte))
        print(f"[EVAL] top1={met['top1']:.3f}  top2={met['top2']:.3f}  logloss={met['logloss']:.3f}")

    else:  # predict
        pack = pickle.load(open(args.model,"rb")); mdl = pack["model"]
        row = { "pitcher_id": args.pitcher_id, "stand": args.stand, "p_throws": args.p_throws,
                "balls": args.balls, "strikes": args.strikes, "prev_pitch_type": args.prev_pitch_type.upper(),
                "on_1b": args.on_1b, "on_2b": args.on_2b, "on_3b": args.on_3b, "tto_bucket": args.tto_bucket }
        p = mdl.predict_proba(pd.DataFrame([row]))[0]
        order = np.argsort(-p)
        print("Next pitch type probabilities:")
        for i in order: print(f"  {PITCH_CANON[i]:>2}: {p[i]:.1%}")
        print("Top‑2:", [PITCH_CANON[i] for i in order[:2]])
if __name__=="__main__": main()
