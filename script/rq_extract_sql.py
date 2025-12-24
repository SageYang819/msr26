# rq_extract_sql.py
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine, text


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

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

JOBS = [
    ("pr_scenarios_rq1.sql", "pr_scenarios_rq1", "rq1_pr_scenarios"),
    ("pr_cost_rq2.sql", "pr_cost_rq2", "rq2_pr_cost"),
    ("RQ23.sql", "pr_rq12", "rq23"),
]

def read_sql_file(fp: str) -> str:
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(f"Missing SQL file: {p.resolve()}")
    return p.read_text(encoding="utf-8")

def split_sql(sql_text: str):
    stmts = []
    for s in sql_text.split(";"):
        s = s.strip()
        if not s:
            continue
        lines = []
        for line in s.splitlines():
            t = line.strip()
            if t.startswith("--"):
                continue
            lines.append(line)
        stmt = "\n".join(lines).strip()
        if stmt:
            stmts.append(stmt)
    return stmts

def should_skip(stmt: str) -> bool:
    s = stmt.strip().lower()
    return s.startswith("create database") or s.startswith("use ")

def exec_sql_allow_duplicates(conn, stmt: str):
    try:
        conn.exec_driver_sql(stmt)
    except Exception as e:
        msg = str(e)
        # 1061 duplicate index name
        if "1061" in msg and "Duplicate key name" in msg:
            return
        # 1007 database exists
        if "1007" in msg and "database exists" in msg:
            return
        raise

with engine.begin() as conn:
    for sql_file, obj_name, out_stem in JOBS:
        sql_text = read_sql_file(sql_file)
        print("running:", sql_file)

        for stmt in split_sql(sql_text):
            if should_skip(stmt):
                continue
            exec_sql_allow_duplicates(conn, stmt)

        df = pd.read_sql_query(text(f"SELECT * FROM {obj_name}"), conn)

        out_path = OUT_DIR / f"{out_stem}.csv"
        df.to_csv(out_path, index=False)
        print(out_stem, "shape=", df.shape, "saved=", out_path.resolve())

engine.dispose()
print("Extract done.")
