from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

metrics_path = OUT_DIR / "rq3_topk_metrics_repeated_repo_split.csv"
summary_path = OUT_DIR / "rq3_topk_metrics_summary.csv"

if not metrics_path.exists():
    raise FileNotFoundError(f"Missing: {metrics_path.resolve()}")
if not summary_path.exists():
    raise FileNotFoundError(f"Missing: {summary_path.resolve()}")

m = pd.read_csv(metrics_path)
s = pd.read_csv(summary_path)

# Plot 1: mean±std P/R/F1 by budget (two lines per metric: agent_prior vs logistic)
for metric in ["precision", "recall", "f1"]:
    fig, ax = plt.subplots()
    for model in s["model"].unique():
        sub = s[s["model"] == model].sort_values("budget_frac")
        y = sub[f"{metric}_mean"].values
        yerr = sub[f"{metric}_std"].values
        ax.errorbar(sub["budget_frac"].values, y, yerr=yerr, marker="o", capsize=3, label=model)
    ax.set_xlabel("Top-k budget fraction")
    ax.set_ylabel(f"{metric.upper()} (mean±std)")
    ax.legend()
    plt.tight_layout()
    out_fig = OUT_DIR / f"fig_rq3_{metric}_budget_mean_std.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", out_fig.resolve())

# Plot 2: distribution across splits (boxplot) for F1
fig, ax = plt.subplots()

# short x labels
short = {
    "agent_prior": "AP",
    "logit_agent_plus_early": "LR",
}

labels = []
data = []
for frac in sorted(m["budget_frac"].unique()):
    for model in ["agent_prior", "logit_agent_plus_early"]:
        sub = m[(m["budget_frac"] == frac) & (m["model"] == model)]
        labels.append(f"{short.get(model, model)}-{int(round(frac * 100))}%")
        data.append(sub["f1"].values)

ax.boxplot(data, labels=labels, vert=True)
ax.set_ylabel("F1 across repo-splits")
plt.tight_layout()

out_box = OUT_DIR / "fig_rq3_f1_boxplot_by_budget.png"
plt.savefig(out_box, dpi=300, bbox_inches="tight")
plt.close()
print("saved:", out_box.resolve())

# Tables already exist; just print paths for convenience
print("saved:", summary_path.resolve())
print("saved:", metrics_path.resolve())

