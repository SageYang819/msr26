from pathlib import Path
import os
import math

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import statsmodels.api as sm
import statsmodels.formula.api as smf


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

q = float(os.getenv("AIDEV_HIGH_COST_Q", "0.75"))
abs_thr = os.getenv("AIDEV_HIGH_COST_THR", "").strip()
min_n = int(os.getenv("AIDEV_MIN_N", "50"))
top_k = int(os.getenv("AIDEV_TOP_K", "10"))

if abs_thr:
    thr = float(abs_thr)
else:
    thr = float(df["total_cost"].quantile(q))

df["is_high_cost"] = (df["total_cost"] >= thr).astype(int)

overall_n = int(df.shape[0])
overall_rate = float(df["is_high_cost"].mean())

g = df.groupby(["agent", "scenario_label"], as_index=False).agg(
    n=("pr_id", "size"),
    high_cost_n=("is_high_cost", "sum"),
)
g["high_cost_rate"] = g["high_cost_n"] / g["n"]
g["lift_vs_overall"] = g["high_cost_rate"] / overall_rate if overall_rate > 0 else np.nan

top = (g[g["n"] >= min_n]
       .sort_values(["high_cost_rate", "n"], ascending=[False, False])
       .head(top_k)
       .copy())

top_out = OUT_DIR / "rq2_top_risk_agent_x_scenario.csv"
top.to_csv(top_out, index=False)

chi_rows = []

def cramers_v(chi2_stat, n, r, c):
    denom = n * max(1, min(r - 1, c - 1))
    return math.sqrt(chi2_stat / denom) if denom > 0 else float("nan")

for scn, sub in df.groupby("scenario_label"):
    tab = pd.crosstab(sub["agent"], sub["is_high_cost"])
    if tab.shape[1] == 1:
        tab[1 - tab.columns[0]] = 0
        tab = tab[[0, 1]] if 0 in tab.columns else tab[[1, 0]]
    chi2_stat, p, dof, _ = chi2_contingency(tab.values)
    n = int(tab.values.sum())
    v = cramers_v(chi2_stat, n, tab.shape[0], tab.shape[1])
    chi_rows.append({
        "scenario_label": scn,
        "n": n,
        "chi2": float(chi2_stat),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "agents_k": int(tab.shape[0]),
    })

chi_df = pd.DataFrame(chi_rows).sort_values("scenario_label")
chi_out = OUT_DIR / "rq2_chi2_by_scenario.csv"
chi_df.to_csv(chi_out, index=False)

tab_combo = pd.crosstab(df["agent"].astype(str) + " | " + df["scenario_label"].astype(str), df["is_high_cost"])
if tab_combo.shape[1] == 1:
    tab_combo[1 - tab_combo.columns[0]] = 0
    tab_combo = tab_combo[[0, 1]] if 0 in tab_combo.columns else tab_combo[[1, 0]]
chi2_combo, p_combo, dof_combo, _ = chi2_contingency(tab_combo.values)
v_combo = cramers_v(chi2_combo, int(tab_combo.values.sum()), tab_combo.shape[0], tab_combo.shape[1])

full = smf.glm(
    "is_high_cost ~ C(agent) * C(scenario_label)",
    data=df,
    family=sm.families.Binomial(),
).fit(cov_type="HC3")

reduced = smf.glm(
    "is_high_cost ~ C(agent) + C(scenario_label)",
    data=df,
    family=sm.families.Binomial(),
).fit(cov_type="HC3")

lr_stat = float(2 * (full.llf - reduced.llf))
df_diff = int(full.df_model - reduced.df_model)
lr_p = float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else float("nan")

coef = full.params.copy()
se = full.bse.copy()
pvals = full.pvalues.copy()

coef_df = pd.DataFrame({
    "term": coef.index,
    "beta": coef.values,
    "se": se.values,
    "p_value": pvals.values,
})
coef_df["or"] = np.exp(coef_df["beta"])
coef_df["or_ci_low_95"] = np.exp(coef_df["beta"] - 1.96 * coef_df["se"])
coef_df["or_ci_high_95"] = np.exp(coef_df["beta"] + 1.96 * coef_df["se"])

coef_out = OUT_DIR / "rq2_logit_coefficients.csv"
coef_df.to_csv(coef_out, index=False)

inter = coef_df[coef_df["term"].str.contains(":", regex=False)].sort_values("p_value")
inter_out = OUT_DIR / "rq2_logit_interactions_top.csv"
inter.head(top_k).to_csv(inter_out, index=False)

def fmt_p(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"

def fmt(x, nd=3):
    return "NA" if pd.isna(x) else f"{x:.{nd}f}"

lines = []
lines.append("RQ2 Stats Summary")
lines.append(f"High-cost definition: total_cost >= {fmt(thr, 3)} (q={fmt(q, 2)}; override_thr={'yes' if abs_thr else 'no'})")
lines.append(f"Sample size: n={overall_n}, overall_high_cost_rate={fmt(overall_rate, 3)}")
lines.append("")
lines.append("Logistic (inference, not prediction)")
lines.append("Model: is_high_cost ~ C(agent) * C(scenario_label)")
lines.append(f"LR test (interaction vs main-effects): LR={fmt(lr_stat, 3)}, df={df_diff}, p={fmt_p(lr_p)}")
lines.append(f"Top interaction terms (by p): saved={inter_out.resolve()}")
lines.append("")
lines.append("Chi-square by scenario (agent x high_cost)")
lines.append(f"Saved: {chi_out.resolve()}")
for _, r in chi_df.iterrows():
    lines.append(
        f"- {r['scenario_label']}: n={int(r['n'])}, chi2={fmt(r['chi2'], 3)}, dof={int(r['dof'])}, "
        f"p={fmt_p(r['p_value'])}, Cramer's V={fmt(r['cramers_v'], 3)}"
    )
lines.append("")
lines.append("Chi-square on (agent|scenario) combos vs high_cost")
lines.append(
    f"- combos_k={tab_combo.shape[0]}, n={int(tab_combo.values.sum())}, chi2={fmt(chi2_combo, 3)}, "
    f"dof={int(dof_combo)}, p={fmt_p(p_combo)}, Cramer's V={fmt(v_combo, 3)}"
)
lines.append("")
lines.append("Top-risk (agent, scenario) combos")
lines.append(f"Filter: n >= {min_n}, top_k={top_k}")
for _, r in top.iterrows():
    lines.append(
        f"- {r['agent']} | {r['scenario_label']}: rate={fmt(r['high_cost_rate'], 3)} "
        f"(high_cost_n={int(r['high_cost_n'])}, n={int(r['n'])}), lift={fmt(r['lift_vs_overall'], 3)}"
    )
lines.append("")
lines.append(f"Saved: {top_out.resolve()}")
lines.append(f"Saved: {coef_out.resolve()}")

summary_out = OUT_DIR / "rq2_stats_summary.txt"
summary_out.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("saved:", summary_out.resolve())
