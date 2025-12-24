# Early Warning of High-Cost Agentic Pull Requests via Scenario–Cost Modeling

Replication package for the MSR 2026 Mining Challenge paper:

> Early Warning of High-Cost Agentic Pull Requests via Scenario–Cost Modeling

This repository contains the code and derived tables/figures needed to reproduce
the results in the paper. All steps are scripted so that a single command
recreates the intermediate PR-level tables and all final outputs.

## 1. Dataset

We use the AIDev dataset (Zenodo v3):

- DOI: `10.5281/zenodo.16919272`

The raw AIDev tables are **not** included in this repository. They must be
downloaded separately from Zenodo (Parquet format).

## 2. Prerequisites

- Conda (Miniconda or Anaconda)
- A local MySQL server that you can connect to (e.g., on `localhost`)

The analysis uses:

- Python 3.11 (via a conda environment)
- A small set of Python packages listed in `requirements.txt`
- MySQL as a local store for the raw AIDev tables

## 3. Environment setup

### 3.1 Create and activate the `msr` environment

```bash
conda create -n msr python=3.11 -y
conda activate msr
cd /Users/young/MSR
```

(If you cloned this repository to a different path, replace `/Users/young/MSR`
with your own project path.)

### 3.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

(Alternatively, you can use `environment.yml` if provided, via
`conda env create -f environment.yml`.)

## 4. Configuration

The scripts expect database and data configuration via environment variables.
A convenient way is to create a `.env` file in the project root (this file is
ignored by git).

**Required variables:**

- `DB_HOST` – MySQL host (e.g., `localhost`)
- `DB_PORT` – MySQL port (e.g., `3306`)
- `DB_USER` – MySQL user
- `DB_PASS` – MySQL password
- `DB_NAME` – Database name to use for AIDev tables (e.g., `aidev`)
- `DATA_DIR` – Directory containing the AIDev Parquet files downloaded from Zenodo

Example `.env` (do **not** commit this file):

```bash
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASS=your_password
DB_NAME=aidev
DATA_DIR=/path/to/aidev/parquet
```

## 5. How to reproduce all results

Once the environment and `.env` are configured:

```bash
conda activate msr
cd /Users/young/MSR
python -u script/run_all.py
```

This will:

1. Connect to MySQL and import the AIDev Parquet tables into the database  
   (`script/rq_load_mysql.py`).
2. Materialize PR-level tables and a wide PR table using SQL  
   (`script/rq_extract_sql.py`, including `pr_scenarios_rq1.sql`,
   `pr_cost_rq2.sql`, `RQ23.sql`).
3. Run the three analysis stages:
   - RQ1 scripts (`script/rq1_outputs.py`)
   - RQ2 scripts (`script/rq2_outputs.py`, `script/rq2_stats.py`)
   - RQ3 scripts (`script/rq3_train.py`, `script/rq3_outputs.py`,
     `script/rq3_agent_prior_table.py`)
4. Write all derived tables and figures to `script/outputs/`.

All outputs mentioned in the paper are generated automatically by this
pipeline, starting only from the raw AIDev Parquet files and MySQL.

## 6. Script overview

All analysis scripts are under `script/`:

- `run_all.py`  
  Orchestrates the full pipeline (DB import → SQL extraction → RQ1 → RQ2 → RQ3).

- `rq_load_mysql.py`  
  Loads AIDev Parquet tables from `DATA_DIR` into MySQL.

- `rq_extract_sql.py`  
  Runs the SQL scripts:
  - `pr_scenarios_rq1.sql` – builds the PR-level scenario table
    (`rq1_pr_scenarios.csv`)
  - `pr_cost_rq2.sql` – builds the PR-level cost table (`rq2_pr_cost.csv`)
  - `RQ23.sql` – builds a wide PR table (`rq23.csv`)

- `rq1_outputs.py`  
  Constructs RQ1 tables and figures (scenario distribution by agent).

- `rq2_outputs.py`  
  Constructs RQ2 tables and figures (high-cost concentration across
  agent–scenario cells).

- `rq2_stats.py`  
  Runs statistical tests and models for RQ2 (chi-square tests, logistic
  regression with interactions).

- `rq3_train.py`  
  Trains the early-warning models for RQ3 (agent-prior baseline and
  logistic regression ranker).

- `rq3_outputs.py`  
  Generates RQ3 evaluation outputs and figures (Top-k performance under
  repeated repo-level splits).

- `rq3_agent_prior_table.py`  
  Exports agent-level high-cost rates used by the agent-prior baseline.

## 7. Outputs (by RQ)

All derived outputs are written to:

- `script/outputs/`

### 7.1 Shared / PR-level tables

- `rq1_pr_scenarios.csv`  
  PR-level scenario labels (S0/S1/S2) used by RQ1 and RQ2.
- `rq2_pr_cost.csv`  
  PR-level cost and high-cost indicator used by RQ2 and RQ3.
- `rq23.csv`  
  Wide PR table used for downstream analyses.

### 7.2 RQ1 – Collaboration scenarios

- `fig_scenario_share_by_agent.png`  
  Scenario share (S0/S1/S2) by agent.
- `table_counts_agent_x_scenario.csv`  
- `table_share_agent_x_scenario.csv`  

These tables correspond to the scenario distribution by agent.

### 7.3 RQ2 – Maintainer effort and risk concentration

Key tables:

- `table_high_cost_agent_x_scenario_long.csv`
- `table_high_cost_rate_agent_x_scenario.csv`
- `table_high_cost_count_agent_x_scenario.csv`
- `table_n_agent_x_scenario.csv`
- `table_high_cost_by_agent.csv`
- `table_top_risk_agent_x_scenario.csv`
- `rq2_chi2_by_scenario.csv`
- `rq2_logit_coefficients.csv`
- `rq2_logit_interactions_top.csv`
- `rq2_top_risk_agent_x_scenario.csv`
- `rq2_stats_summary.txt`

Key figures:

- `fig_high_cost_rate_heatmap_agent_x_scenario.png`  
  High-cost rate heatmap by agent × scenario cell.
- `fig_high_cost_by_agent.png`  
  High-cost rates summarized by agent.

These outputs support the RQ2 findings on cost concentration and the statistical
evidence for agent–scenario interactions.

### 7.4 RQ3 – Early warning under fixed review budget

Tables:

- `rq3_agent_prior_high_cost_rate.csv`
- `rq3_topk_metrics_repeated_repo_split.csv`
- `rq3_topk_metrics_summary.csv`
- `rq3_summary_topk_mean_std.txt`

Figures:

- `fig_rq3_precision_budget_mean_std.png`
- `fig_rq3_recall_budget_mean_std.png`
- `fig_rq3_f1_budget_mean_std.png`
- `fig_rq3_f1_boxplot_by_budget.png`

These outputs support the RQ3 analysis of fixed-budget Top-k prioritization and
the performance of the agent-prior baseline and logistic regression ranker under
repeated repo-level splits.

## 8. Notes on anonymity (for submission)

This repository is hosted under a personal GitHub account and therefore
reveals author identity. For anonymous submission, do **not** include this
repository’s URL in the paper. Instead, describe the replication package
at a high level (as in the paper’s “How to Access” section) and add the
actual link after the review process, once anonymity is no longer required.
