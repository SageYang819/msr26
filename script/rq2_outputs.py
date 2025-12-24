from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RQ1_PATH = OUT_DIR / "rq1_pr_scenarios.csv"
RQ2_PATH = OUT_DIR / "rq2_pr_cost.csv"

if not RQ1_PATH.exists():
    raise FileNotFoundError(f"Missing: {RQ1_PATH.resolve()}")
if not RQ2_PATH.exists():
    raise FileNotFoundError(f"Missing: {RQ2_PATH.resolve()}")

def pick_col(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

rq1 = pd.read_csv(RQ1_PATH)
rq2 = pd.read_csv(RQ2_PATH)

rq1_id = pick_col(rq1.columns, ["pr_id", "id", "pull_request_id"])
rq1_agent = pick_col(rq1.columns, ["agent"])
rq1_scn = pick_col(rq1.columns, ["scenario_label", "scenario"])

rq2_id = pick_col(rq2.columns, ["pr_id", "id", "pull_request_id"])

if not rq1_id or not rq1_agent or not rq1_scn:
    raise ValueError(f"rq1_pr_scenarios missing cols. got={rq1.columns.tolist()}")
if not rq2_id:
    raise ValueError(f"rq2_pr_cost missing id col. got={rq2.columns.tolist()}")

rq1 = rq1.rename(columns={rq1_id: "pr_id", rq1_agent: "agent", rq1_scn: "scenario_label"})
rq2 = rq2.rename(columns={rq2_id: "pr_id"})

cost_cols = ["review_count", "request_changes_count", "comment_count", "post_review_review_count"]
missing_cost = [c for c in cost_cols if c not in rq2.columns]
if missing_cost:
    raise ValueError(f"rq2_pr_cost missing cost cols: {missing_cost}")

for c in cost_cols:
    rq2[c] = pd.to_numeric(rq2[c], errors="coerce").fillna(0)

df = rq1[["pr_id", "agent", "scenario_label"]].merge(
    rq2[["pr_id"] + cost_cols],
    on="pr_id",
    how="inner",
)

df["total_cost"] = df[cost_cols].sum(axis=1)

q = float(os.getenv("AIDEV_HIGH_COST_Q", "0.75"))          # quantile threshold
abs_thr = os.getenv("AIDEV_HIGH_COST_THR", "").strip()     # optional override
min_n = int(os.getenv("AIDEV_MIN_N", "50"))                # for top-risk listing

if abs_thr:
    thr = float(abs_thr)
else:
    thr = float(df["total_cost"].quantile(q))

df["is_high_cost"] = (df["total_cost"] >= thr).astype(int)

g = df.groupby(["agent", "scenario_label"], as_index=False).agg(
    n=("pr_id", "size"),
    high_cost_n=("is_high_cost", "sum"),
)
g["high_cost_rate"] = g["high_cost_n"] / g["n"]

g.to_csv(OUT_DIR / "table_high_cost_agent_x_scenario_long.csv", index=False)

scenario_order = ["S0_Solo_agent", "S1_Human_reviewed", "S2_Human_coedited"]
pivot_rate = g.pivot(index="agent", columns="scenario_label", values="high_cost_rate").fillna(0)
pivot_cnt = g.pivot(index="agent", columns="scenario_label", values="high_cost_n").fillna(0)
pivot_n = g.pivot(index="agent", columns="scenario_label", values="n").fillna(0)

for c in scenario_order:
    if c not in pivot_rate.columns:
        pivot_rate[c] = 0.0
    if c not in pivot_cnt.columns:
        pivot_cnt[c] = 0.0
    if c not in pivot_n.columns:
        pivot_n[c] = 0.0

pivot_rate = pivot_rate[scenario_order]
pivot_cnt = pivot_cnt[scenario_order]
pivot_n = pivot_n[scenario_order]

agent_order = pivot_rate.mean(axis=1).sort_values(ascending=False).index
pivot_rate = pivot_rate.loc[agent_order]
pivot_cnt = pivot_cnt.loc[agent_order]
pivot_n = pivot_n.loc[agent_order]

pivot_rate.reset_index().to_csv(OUT_DIR / "table_high_cost_rate_agent_x_scenario.csv", index=False)
pivot_cnt.reset_index().to_csv(OUT_DIR / "table_high_cost_count_agent_x_scenario.csv", index=False)
pivot_n.reset_index().to_csv(OUT_DIR / "table_n_agent_x_scenario.csv", index=False)

by_agent = df.groupby("agent", as_index=False).agg(
    n=("pr_id", "size"),
    high_cost_n=("is_high_cost", "sum"),
)
by_agent["high_cost_rate"] = by_agent["high_cost_n"] / by_agent["n"]
by_agent = by_agent.sort_values("high_cost_rate", ascending=False)
by_agent.to_csv(OUT_DIR / "table_high_cost_by_agent.csv", index=False)

top = g[g["n"] >= min_n].sort_values(["high_cost_rate", "n"], ascending=[False, False]).copy()
top.to_csv(OUT_DIR / "table_top_risk_agent_x_scenario.csv", index=False)

heat = pivot_rate.values
fig, ax = plt.subplots()
im = ax.imshow(heat, aspect="auto")
ax.set_yticks(range(len(pivot_rate.index)))
ax.set_yticklabels(pivot_rate.index.tolist())
ax.set_xticks(range(len(pivot_rate.columns)))
ax.set_xticklabels(pivot_rate.columns.tolist(), rotation=35, ha="right")
ax.set_xlabel("Scenario")
ax.set_ylabel("Agent")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("High-cost rate")
plt.tight_layout()
out_heat = OUT_DIR / "fig_high_cost_rate_heatmap_agent_x_scenario.png"
plt.savefig(out_heat, dpi=300, bbox_inches="tight")
plt.close()

ax = by_agent.set_index("agent")["high_cost_rate"].plot(kind="bar")
ax.set_ylabel("High-cost rate")
ax.set_xlabel("Agent")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
out_bar = OUT_DIR / "fig_high_cost_by_agent.png"
plt.savefig(out_bar, dpi=300, bbox_inches="tight")
plt.close()

print("thr_total_cost:", thr)
print("saved:", (OUT_DIR / "table_high_cost_agent_x_scenario_long.csv").resolve())
print("saved:", (OUT_DIR / "table_high_cost_rate_agent_x_scenario.csv").resolve())
print("saved:", (OUT_DIR / "table_high_cost_count_agent_x_scenario.csv").resolve())
print("saved:", (OUT_DIR / "table_n_agent_x_scenario.csv").resolve())
print("saved:", (OUT_DIR / "table_high_cost_by_agent.csv").resolve())
print("saved:", (OUT_DIR / "table_top_risk_agent_x_scenario.csv").resolve())
print("saved:", out_heat.resolve())
print("saved:", out_bar.resolve())
