# rq_load_mysql.py
import os
import time
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine


DB_USER = os.getenv("AIDEV_DB_USER", "root")
DB_PASS = os.getenv("AIDEV_DB_PASS")
DB_HOST = os.getenv("AIDEV_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("AIDEV_DB_PORT", "3306"))
DB_NAME = os.getenv("AIDEV_DB_NAME", "aidev")

DATA_DIR = Path(os.getenv("AIDEV_DATA_DIR", "data")).expanduser().resolve()
IF_EXISTS = os.getenv("AIDEV_IF_EXISTS", "replace")  # replace/append/fail

if not DB_PASS:
    raise ValueError("Missing AIDEV_DB_PASS")

server_engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/?charset=utf8mb4",
    pool_pre_ping=True,
)

with server_engine.begin() as conn:
    conn.exec_driver_sql(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
    pool_pre_ping=True,
)

tables = [
    "pull_request",
    "pr_comments",
    "pr_reviews",
    "pr_review_comments_v2",
    "pr_commits",
    "pr_timeline",
    "user",
]
name_map = {"user": "aidev_user"}

DROP_COLS = {
    "body", "diff", "patch", "description", "message", "text", "content",
    "diff_hunk", "hunk", "raw_diff", "raw_patch", "code", "snippet"
}

missing = [str(DATA_DIR / f"{t}.parquet") for t in tables if not (DATA_DIR / f"{t}.parquet").exists()]
if missing:
    raise FileNotFoundError("Missing parquet:\n" + "\n".join(missing))

with engine.connect() as conn:
    print("DB test:", conn.exec_driver_sql("SELECT 1").fetchone())

print("DATA_DIR:", DATA_DIR)
print("IF_EXISTS:", IF_EXISTS)

for t in tables:
    t0 = time.time()

    path = DATA_DIR / f"{t}.parquet"
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")

    target = name_map.get(t, t)

    print(f"\n{t} -> {target}")
    print("shape:", df.shape)

    df.to_sql(
        target,
        con=engine,
        if_exists=IF_EXISTS,
        index=False,
        chunksize=50_000,
        method="multi",
    )

    print(f"done: {time.time() - t0:.1f}s")

engine.dispose()
server_engine.dispose()
print("Import done.")
