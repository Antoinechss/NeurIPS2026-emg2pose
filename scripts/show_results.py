"""Print the public scores of the latest benchopt runs.

Usage: python scripts/show_results.py [n_runs]   (reads $B/outputs, default 1 run)
"""

import os
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["solver_name", "objective_angular_mae_public",
           "objective_angular_mae_subject_std_public"]

outputs = Path(os.environ["B"]) / "outputs"
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for path in sorted(outputs.glob("*.parquet"))[-n_runs:]:
    df = pd.read_parquet(path)
    print(path.name)
    print(df[[c for c in COLUMNS if c in df]].to_string(index=False))
