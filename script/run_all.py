# run_all.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)

def main():
    run([sys.executable, "rq_load_mysql.py"])
    run([sys.executable, "rq_extract_sql.py"])

    run([sys.executable, "rq1_outputs.py"])

    run([sys.executable, "rq2_outputs.py"])
    run([sys.executable, "rq2_stats.py"])

    run([sys.executable, "rq3_train.py"])
    run([sys.executable, "rq3_outputs.py"])
    run([sys.executable, "rq3_agent_prior_table.py"])

    print("\nALL DONE")

if __name__ == "__main__":
    main()
