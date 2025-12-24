from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DB_USER = os.getenv("AIDEV_DB_USER", "root")
DB_PASS = os.getenv("AIDEV_DB_PASS")
DB_HOST = os.getenv("AIDEV_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("AIDEV_DB_PORT", "3306"))
DB_NAME = os.getenv("AIDEV_DB_NAME", "aidev")

if not DB_PASS:
    raise ValueError("Missing AIDEV_DB_PASS")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
    pool_pre_ping=True,
)

df = pd.read_sql(
    """
    SELECT agent, scenario_label, COUNT(*) AS n
    FROM pr_scenarios_rq1
    GROUP BY agent, scenario_label;
    """,
    engine,
)

pivot = df.pivot(index="agent", columns="scenario_label", values="n").fillna(0)

cols = ["S0_Solo_agent", "S1_Human_reviewed", "S2_Human_coedited"]
missing = [c for c in cols if c not in pivot.columns]
if missing:
    raise ValueError(f"Missing scenario_label columns: {missing}")

pivot = pivot[cols]
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

share = pivot.div(pivot.sum(axis=1), axis=0)

ax = share.plot(kind="bar", stacked=True)
ax.set_ylabel("Proportion of PRs")
ax.set_xlabel("Agent")
ax.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

out_path = OUT_DIR / "fig_scenario_share_by_agent.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("saved:", out_path.resolve())

table_counts = pivot.copy()
table_counts.to_csv(OUT_DIR / "table_counts_agent_x_scenario.csv")

table_share = share.copy()
table_share.to_csv(OUT_DIR / "table_share_agent_x_scenario.csv")

print("saved:", (OUT_DIR / "table_counts_agent_x_scenario.csv").resolve())
print("saved:", (OUT_DIR / "table_share_agent_x_scenario.csv").resolve())
