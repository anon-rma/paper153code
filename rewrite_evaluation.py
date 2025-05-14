import os
import pandas as pd
from Levenshtein import distance as lev_dist  # pip install python-Levenshtein

import re
import spacy
nlp = spacy.load("en_core_web_sm")
# python -m spacy download en_core_web_sm
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
    return 1.0 if ref == hyp else 0.0

def evaluate_sentence_pair(A, B):    
    dist = lev_dist(A, B)
    if dist <= 3:
        return {"final_correct": True, "details": {"semantic_distance": dist}}    
    phones_a, phones_b = extract_phones(A), extract_phones(B)
    uris_a,   uris_b   = extract_uris(A),   extract_uris(B)
    emails_a, emails_b = extract_emails(A), extract_emails(B)
    addr_a,   addr_b   = extract_addresses(A), extract_addresses(B)
    noun_a,   noun_b   = extract_nouns(A),   extract_nouns(B)

    checks = [
        exact_match_score(phones_a, phones_b),
        exact_match_score(uris_a,   uris_b),
        exact_match_score(emails_a, emails_b),
        exact_match_score(addr_a,   addr_b),
        exact_match_score(noun_a,   noun_b),
    ]
    return {"final_correct": all(checks), "details": {"semantic_distance": dist}}

base_dir = "datasets/tc"
rew_dir  = os.path.join(base_dir, "Qwen3-1.7b-half_rewrited") # Phi-half_rewrited

file_list = [
    fn for fn in os.listdir(base_dir)
    if fn != "it2_NR_tc.tsv" and fn.endswith(".tsv") and os.path.isfile(os.path.join(base_dir, fn))
]

results = []

for file_name in sorted(file_list):
    path_a = os.path.join(base_dir, file_name)
    path_b = os.path.join(rew_dir,  file_name)

    if not os.path.exists(path_b):
        continue

    df_a = pd.read_csv(path_a, sep="\t", dtype=str)
    df_b = pd.read_csv(path_b, sep="\t", dtype=str)

    if len(df_a) != len(df_b):
        print(f"[에러] '{file_name}' 행 수 불일치: {len(df_a)} vs {len(df_b)}")
        continue

    total = len(df_a)
    correct_cnt = 0

    for a, b in zip(df_a["rewrited_query"].fillna(""), df_b["rewrited_query"].fillna("")):
        result = evaluate_sentence_pair(a, b)
        if result["final_correct"]:
            correct_cnt += 1

    acc_pct = correct_cnt / total * 100

    print(f"{file_name} -> Accuracy (final_correct 기준): {acc_pct:.2f}%")

    results.append({
        "file": file_name,
        "Accuracy (%)": round(acc_pct, 2)
    })

results_df = pd.DataFrame(results).set_index("file")
results_df.loc["Average"] = results_df.mean()

avg_acc = results_df.loc["Average", "Accuracy (%)"]
print(f"\naverage: {avg_acc:.2f}%")
results_df.to_csv("rma_logs/rewrited_score.tsv", sep="\t")
