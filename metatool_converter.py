import pandas as pd
import json
import random

data_path   = "all_clean_data.csv"
desc_json  = "apis/plugin_des.json"
output_csv = "metatool_testset.tsv"
random.seed(42)

df = pd.read_csv(data_path)
with open(desc_json, "r", encoding="utf-8") as f:
    desc_map = json.load(f)

all_tools = df["Tool"].unique().tolist()

sampled = (
    df.groupby("Tool", group_keys=False)
      .head(5)
      .copy()
)

def build_candidates(correct_tool: str) -> list[str]:
    others = [t for t in all_tools if t != correct_tool]
    candidates = random.sample(others, 4) + [correct_tool]
    random.shuffle(candidates)
    return candidates

sampled["candidates"] = sampled["Tool"].apply(build_candidates)
sampled.rename(columns={"Tool": "answer"}, inplace=True)
sampled["answer"] = sampled["answer"].apply(
    lambda t: {"plan": t, "arguments": {}}
)

sampled = sampled.drop(columns=[c for c in sampled.columns if c.lower() == "description"], errors="ignore")
sampled.to_csv(output_csv, sep="\t", index=False)
print(f"Saved {len(sampled)} rows → {output_csv}")
