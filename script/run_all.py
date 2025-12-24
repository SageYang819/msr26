from pathlib import Path
import shutil

def _collect_outputs():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "script" / "outputs"
    data_dir = root / "data"
    plots_dir = root / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # CSV -> data/
    for p in out_dir.glob("*.csv"):
        shutil.move(str(p), str(data_dir / p.name))

    # TXT -> data/
    for p in out_dir.glob("*.txt"):
        shutil.move(str(p), str(data_dir / p.name))

    # PNG -> plots/
    for p in out_dir.glob("*.png"):
        shutil.move(str(p), str(plots_dir / p.name))
