from pathlib import Path
import os
import pandas as pd

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RQ23_PATH = OUT_DIR / "rq23.csv"
RQ1_PATH = OUT_DIR / "rq1_pr_scenarios.csv"
RQ2_PATH = OUT_DIR / "rq2_pr_cost.csv"

HIGH_COST_Q = float(os.getenv("AIDEV_HIGH_COST_Q", "0.75"))

def pick_col(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def ensure_pr_id(df):
    cid = pick_col(df.columns, ["pr_id", "pull_request_id", "id"])
    if not cid:
        raise ValueError(f"Missing pr id col. got={df.columns.tolist()}")
    if cid != "pr_id":
        df = df.rename(columns={cid: "pr_id"})
    return df

rq23 = pd.read_csv(RQ23_PATH)
rq23 = ensure_pr_id(rq23)

rq1 = pd.read_csv(RQ1_PATH)
rq1 = ensure_pr_id(rq1)
if "agent" not in rq1.columns:
    raise ValueError(f"Missing agent in rq1. got={rq1.columns.tolist()}")

rq2 = pd.read_csv(RQ2_PATH)
rq2 = ensure_pr_id(rq2)
cost_cols = ["review_count", "request_changes_count", "comment_count", "post_review_review_count"]
for c in cost_cols:
    rq2[c] = pd.to_numeric(rq2[c], errors="coerce").fillna(0)

df = (rq23[["pr_id"]]
      .merge(rq1[["pr_id", "agent"]], on="pr_id", how="left")
      .merge(rq2[["pr_id"] + cost_cols], on="pr_id", how="left"))

df["total_cost"] = df[cost_cols].sum(axis=1)
thr = float(df["total_cost"].quantile(HIGH_COST_Q))
df["is_high_cost"] = (df["total_cost"] >= thr).astype(int)

prior = (df.groupby("agent", as_index=False)
           .agg(n=("pr_id", "size"), high_cost_rate=("is_high_cost", "mean"))
           .sort_values("high_cost_rate", ascending=False))

out_path = OUT_DIR / "rq3_agent_prior_high_cost_rate.csv"
prior.to_csv(out_path, index=False)

print("thr_total_cost:", thr)
print("saved:", out_path.resolve())
