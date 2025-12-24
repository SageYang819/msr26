import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RQ23_PATH = OUT_DIR / "rq23.csv"
RQ1_PATH = OUT_DIR / "rq1_pr_scenarios.csv"
RQ2_PATH = OUT_DIR / "rq2_pr_cost.csv"

BUDGET_FRACS = [0.10, 0.20]
N_SPLITS = int(os.getenv("RQ3_N_SPLITS", "30"))
TEST_SIZE = float(os.getenv("RQ3_TEST_SIZE", "0.20"))
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


def ensure_agent(df):
    c = pick_col(df.columns, ["agent"])
    if not c:
        raise ValueError(f"Missing agent col. got={df.columns.tolist()}")
    if c != "agent":
        df = df.rename(columns={c: "agent"})
    return df


def ensure_repo(df):
    c = pick_col(
        df.columns,
        ["repo_id", "repository_id", "repo", "repo_name", "repo_full_name", "full_name"],
    )
    if not c:
        raise ValueError(
            "Missing repo id/name col for repo-level split. "
            f"got={df.columns.tolist()}"
        )
    if c != "repo_id":
        df = df.rename(columns={c: "repo_id"})
    return df


def add_scenario_if_any(df):
    c = pick_col(df.columns, ["scenario_label", "scenario"])
    if c and c != "scenario_label":
        df = df.rename(columns={c: "scenario_label"})
    return df


def compute_high_cost_label(df):
    if "is_high_cost" in df.columns:
        return pd.to_numeric(df["is_high_cost"], errors="coerce").fillna(0).astype(int)

    if "total_cost" in df.columns:
        total = pd.to_numeric(df["total_cost"], errors="coerce").fillna(0)
        thr = float(total.quantile(HIGH_COST_Q))
        return (total >= thr).astype(int)

    cost_cols = ["review_count", "request_changes_count", "comment_count", "post_review_review_count"]
    if all(c in df.columns for c in cost_cols):
        tmp = df[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        total = tmp.sum(axis=1)
        thr = float(total.quantile(HIGH_COST_Q))
        return (total >= thr).astype(int)

    return None


def build_dataset():
    if not RQ23_PATH.exists():
        raise FileNotFoundError(f"Missing: {RQ23_PATH.resolve()}")

    base = pd.read_csv(RQ23_PATH)
    base = ensure_pr_id(base)

    need_agent = "agent" not in base.columns
    need_repo = pick_col(
        base.columns,
        ["repo_id", "repository_id", "repo", "repo_name", "repo_full_name", "full_name"],
    ) is None

    if need_agent or need_repo:
        if not RQ1_PATH.exists():
            raise FileNotFoundError(f"Missing (needed for agent/repo): {RQ1_PATH.resolve()}")
        rq1 = pd.read_csv(RQ1_PATH)
        rq1 = ensure_pr_id(rq1)
        rq1 = ensure_agent(rq1)
        rq1 = add_scenario_if_any(rq1)
        rq1 = ensure_repo(rq1)

        keep = ["pr_id", "agent", "repo_id"]
        if "scenario_label" in rq1.columns:
            keep.append("scenario_label")
        base = base.merge(rq1[keep], on="pr_id", how="left")

    base = ensure_agent(base)
    base = ensure_repo(base)
    base = add_scenario_if_any(base)

    y = compute_high_cost_label(base)
    if y is None:
        if not RQ2_PATH.exists():
            raise FileNotFoundError(f"Missing (needed for label): {RQ2_PATH.resolve()}")
        rq2 = pd.read_csv(RQ2_PATH)
        rq2 = ensure_pr_id(rq2)

        cost_cols = ["review_count", "request_changes_count", "comment_count", "post_review_review_count"]
        missing = [c for c in cost_cols if c not in rq2.columns]
        if missing:
            raise ValueError(f"Missing columns in rq2_pr_cost.csv: {sorted(missing)}")

        merged = base.merge(rq2[["pr_id"] + cost_cols], on="pr_id", how="left")
        y = compute_high_cost_label(merged)
        base = merged

    base["is_high_cost"] = y.astype(int)

    numeric_candidates = [
        "changed_files", "files_changed", "num_files",
        "additions", "deletions", "lines_added", "lines_deleted",
        "commits", "num_commits", "commit_count",
        "title_length", "body_length",
    ]
    num_cols = [c for c in numeric_candidates if c in base.columns]
    cat_cols = ["agent"]

    X = base[
        ["pr_id", "repo_id"]
        + (["scenario_label"] if "scenario_label" in base.columns else [])
        + cat_cols
        + num_cols
    ].copy()
    y = base["is_high_cost"].astype(int).copy()

    X["agent"] = X["agent"].astype(str).fillna("NA")
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X, y, num_cols


def _to_pr_id_numeric(pr_id):
    s = pd.Series(pr_id)
    num = pd.to_numeric(s, errors="coerce")
    if num.isna().any():
        num = pd.Series(pd.factorize(s.astype(str))[0], index=s.index)
    return num.astype(int).to_numpy()


def topk_metrics(y_true, scores, pr_id, k):
    k = int(k)
    k = max(1, min(k, len(scores)))

    scores = np.asarray(scores, dtype=float)
    pr_num = _to_pr_id_numeric(pr_id)

    idx_sorted = np.lexsort((pr_num, -scores))  # score desc, pr_id asc
    idx = idx_sorted[:k]

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[idx] = 1

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f1


def agent_prior_scores(agent_series, train_df):
    prior = train_df.groupby("agent")["is_high_cost"].mean().to_dict()
    global_prior = float(train_df["is_high_cost"].mean())
    return agent_series.map(lambda a: prior.get(a, global_prior)).astype(float).values


def main():
    X, y, num_cols = build_dataset()
    df_all = X.copy()
    df_all["is_high_cost"] = y.values

    pre = ColumnTransformer(
        transformers=[
            ("agent", OneHotEncoder(handle_unknown="ignore"), ["agent"]),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    splitter = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)

    rows = []
    split_id = 0
    for train_idx, test_idx in splitter.split(df_all, groups=df_all["repo_id"]):
        split_id += 1
        train = df_all.iloc[train_idx].copy()
        test = df_all.iloc[test_idx].copy()

        base_scores = agent_prior_scores(test["agent"], train)

        pipe.fit(train[["agent"] + num_cols], train["is_high_cost"])
        proba = pipe.predict_proba(test[["agent"] + num_cols])[:, 1]

        for frac in BUDGET_FRACS:
            k = int(round(frac * len(test)))

            p0, r0, f10 = topk_metrics(
                test["is_high_cost"].values, base_scores, test["pr_id"].values, k
            )
            p1, r1, f11 = topk_metrics(
                test["is_high_cost"].values, proba, test["pr_id"].values, k
            )

            rows.append({
                "split": split_id,
                "budget_frac": float(frac),
                "k": int(k),
                "n_test": int(len(test)),
                "pos_rate_test": float(test["is_high_cost"].mean()),
                "model": "agent_prior",
                "precision": float(p0),
                "recall": float(r0),
                "f1": float(f10),
                "num_features_used": int(1 + len(num_cols)),
            })
            rows.append({
                "split": split_id,
                "budget_frac": float(frac),
                "k": int(k),
                "n_test": int(len(test)),
                "pos_rate_test": float(test["is_high_cost"].mean()),
                "model": "logit_agent_plus_early",
                "precision": float(p1),
                "recall": float(r1),
                "f1": float(f11),
                "num_features_used": int(1 + len(num_cols)),
            })

    res = pd.DataFrame(rows)
    out_metrics = OUT_DIR / "rq3_topk_metrics_repeated_repo_split.csv"
    res.to_csv(out_metrics, index=False)

    summary = (
        res.groupby(["model", "budget_frac"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            splits=("split", "nunique"),
        )
    )
    out_summary = OUT_DIR / "rq3_topk_metrics_summary.csv"
    summary.to_csv(out_summary, index=False)

    lines = []
    lines.append("RQ3 Early Warning (Top-k) Summary")
    lines.append(f"Splits: {N_SPLITS}, test_size: {TEST_SIZE}")
    lines.append(f"Budgets: {BUDGET_FRACS}")
    lines.append(f"Label: high_cost = top {int((1-HIGH_COST_Q)*100)}% by total_cost (q={HIGH_COST_Q})")
    lines.append(f"Predictors: agent + early_numeric={num_cols}")
    lines.append("Tie-break: score desc, pr_id asc")
    lines.append("")
    for _, r in summary.sort_values(["budget_frac", "model"]).iterrows():
        lines.append(
            f"- {r['model']} @ {r['budget_frac']:.2f}: "
            f"P={r['precision_mean']:.3f}±{r['precision_std']:.3f}, "
            f"R={r['recall_mean']:.3f}±{r['recall_std']:.3f}, "
            f"F1={r['f1_mean']:.3f}±{r['f1_std']:.3f} (splits={int(r['splits'])})"
        )

    out_txt = OUT_DIR / "rq3_summary_topk_mean_std.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("saved:", out_metrics.resolve())
    print("saved:", out_summary.resolve())
    print("saved:", out_txt.resolve())


if __name__ == "__main__":
    main()
