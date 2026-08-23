import shutil
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

for csv_file in sorted(folder.glob("fold_*.csv")):
    shutil.copy(csv_file, csv_file.with_suffix(".csv.backup"))
    df = pd.read_csv(csv_file)
    df["id_epitope"] = df["id"]
    df["id_hla"] = df["id"]
    df.to_csv(csv_file, index=False)
    print(f"Processed {csv_file.name}")
