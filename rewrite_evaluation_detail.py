import os
import pandas as pd
from Levenshtein import distance as lev_dist
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_phones(text):
    return set(re.findall(r'\b\d{2,3}-\d{3,4}-\d{4}\b', text))

def extract_uris(text):
    return set(re.findall(r'https?://\S+', text))

def extract_emails(text):
    return set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text))

def extract_addresses(text):
    addr_pattern = r'\d{1,5}\s+[A-Za-z0-9.\s]+\s+(Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr)\b'
    return set(re.findall(addr_pattern, text, flags=re.I))

def extract_nouns(text):
    doc = nlp(text)
    return set(token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"})

def exact_match_score(ref, hyp):
    return ref == hyp

def evaluate_sentence_pair(A, B):
    dist = lev_dist(A, B)
    if dist <= 3:
        return {"final_correct": True, "failed_check": None}

    check_functions = [
        ("extract_phones", extract_phones),
        ("extract_uris", extract_uris),
        ("extract_emails", extract_emails),
        ("extract_addresses", extract_addresses),
        ("extract_nouns", extract_nouns),
    ]

    for func_name, func in check_functions:
        if not exact_match_score(func(A), func(B)):
            return {"final_correct": False, "failed_check": func_name}

    return {"final_correct": True, "failed_check": None}

base_dir = "datasets/tc"
rew_dir  = os.path.join(base_dir, "phi-integrated-half_rewrited")

file_list = [
    fn for fn in os.listdir(base_dir)
    if fn.endswith(".tsv") and os.path.isfile(os.path.join(base_dir, fn))
]

results = []
detailed_results = []

for file_name in sorted(file_list):
    path_a = os.path.join(base_dir, file_name)
    path_b = os.path.join(rew_dir, file_name)

    if not os.path.exists(path_b):
        continue

    df_a = pd.read_csv(path_a, sep="\t", dtype=str)
    df_b = pd.read_csv(path_b, sep="\t", dtype=str)

    if len(df_a) != len(df_b):
        print(f"[Error] '{file_name}' row count mismatch: {len(df_a)} vs {len(df_b)}")
        continue

    total = len(df_a)
    correct_cnt = 0

    row_results = []
    for idx, (a, b) in enumerate(zip(df_a["rewrited_query"].fillna(""), df_b["rewrited_query"].fillna(""))):
        result = evaluate_sentence_pair(a, b)
        is_correct = result["final_correct"]
        if is_correct:
            correct_cnt += 1

        row_results.append({
            "file": file_name,
            "row": idx,
            "query_A": a,
            "query_B": b,
            "correct": is_correct,
            "failed_check": result["failed_check"]
        })

    acc_pct = correct_cnt / total * 100

    results.append({
        "file": file_name,
        "Accuracy (%)": round(acc_pct, 2)
    })

    detailed_results.extend(row_results)

results_df = pd.DataFrame(results).set_index("file")
results_df.loc["Average"] = results_df.mean()

avg_acc = results_df.loc["Average", "Accuracy (%)"]
print(f"\nAverage Accuracy: {avg_acc:.2f}%")

results_df.to_csv("rma_logs/rewrited_score.tsv", sep="\t")
print("Results saved to 'rma_logs/rewrited_score.tsv'")

detailed_results_df = pd.DataFrame(detailed_results)
detailed_results_df.to_csv("rma_logs/detailed_rewrited_score.tsv", sep="\t", index=False)
print("Detailed row-by-row results saved to 'rma_logs/detailed_rewrited_score.tsv'")
