import pandas as pd
from pathlib import Path

folders = ["train", "rma_train", "tc"]
summary_path = Path("summary.tsv")

records = []

for folder in folders:
    for tsv_path in Path(folder).glob("*.tsv"):        
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        records.append({"filename": tsv_path.name, "count": len(df)})

summary_df = pd.DataFrame(records, columns=["filename", "count"])
summary_df.to_csv(summary_path, sep="\t", index=False)

print(f"summary.tsv written with {len(summary_df)} entries")
