import pandas as pd
import random
import json
import argparse
import pdb
from utils.frequently_used_tools import read_jsonl, save_jsonl, get_arg_parse

def analysis(dataset):
    nr_cnt_for_all = 0
    nr_for_this_plan = 0
    nr_cnt_per_plan = {}
    not_nr_cnt_per_plan = {}
    for data in dataset:
        u_idx = data["unique_idx"]
        this_plan = u_idx.split('-')[-2]
        this_idx = u_idx.split('-')[-1]
        
        plan = data["answer"]["plan"]
        if '_NR' in this_idx:
            nr_for_this_plan += 1
            nr_cnt_per_plan[plan] = nr_cnt_per_plan.get(plan, 0) + 1
        else:
            not_nr_cnt_per_plan[plan] = not_nr_cnt_per_plan.get(plan, 0) + 1
        
        if '_NR' in u_idx:
            nr_cnt_for_all += 1

    print(f"nr_cnt_for_all: {nr_cnt_for_all}")
    print(f"nr_for_this_plan: {nr_for_this_plan}")
    print(f"nr_cnt_per_plan: {nr_cnt_per_plan}")
    print(f"not_nr_cnt_per_plan: {not_nr_cnt_per_plan}")

def rewrited_query_dup_printer(filenames):
    all_data = []
    for filename in filenames:
        filename = filename.split(".")[0] + "_dedup.jsonl"
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                data["source_file"] = filename 
                all_data.append(data)

    total_data_count = len(all_data)    
    rewrited_query_counts = {}
    for item in all_data:
        query = item.get("rewrited_query")
        if query is None:
            continue
        rewrited_query_counts[query] = rewrited_query_counts.get(query, 0) + 1

    duplicate_groups = {query: count for query, count in rewrited_query_counts.items() if count > 1}
    duplicate_group_count = len(duplicate_groups)
    duplicate_data_count = sum(duplicate_groups.values())


def rewrited_query_dup_checker(filenames):
    all_data = []
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                data["source_file"] = filename
                all_data.append(data)
    unique_records = {}
    nr_records = []

    for data in all_data:
        u_idx = data.get("unique_idx", "")
        if u_idx.endswith("_NR"):
            nr_records.append(data)
            continue

        key = data.get("rewrited_query")
        if key is None:            
            unique_records[id(data)] = data
        else:
            if key not in unique_records:
                unique_records[key] = data
    final_data = list(unique_records.values()) + nr_records
    total_data_count = len(all_data)
    final_count = len(final_data)
    removed_count = total_data_count - final_count

    print(total_data_count)
    print(final_count)
    print(removed_count)
    grouped_by_file = {}
    for data in final_data:
        source = data["source_file"]
        grouped_by_file.setdefault(source, []).append(data)

    for source_file, records in grouped_by_file.items():
        output_filename = f"{source_file.split('.')[0]}_dedup.jsonl"
        with open(output_filename, "w", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

def weighted_split_dataset(records, ratio=0.9, turn_idx=None):    
    if len(records) == 1:
        return [], records
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if turn_idx == 2:
        weighted_count = 140
    elif turn_idx == 3:
        weighted_count = 30
    elif turn_idx == 4:
        weighted_count = 20
    elif turn_idx == 5:
        weighted_count = 10
    if len(train_records) > weighted_count: 
        train_records = train_records[:weighted_count]
        tc_records = tc_records[:50]
        
    return train_records, tc_records

def split_dataset(records, ratio=0.9):
    random.shuffle(records)
    if len(records) == 1:
        return [], records
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if len(train_records) > 50:
        train_records = train_records[:50]
        tc_records = tc_records[:50]
        
    return train_records, tc_records


def data_spliter(datas, out_file):
    apis = read_jsonl("apis/api_v3.0.2.jsonl")
    apis_keys = set([api["plan"] for api in apis])
    nr_groups = {}       
    non_nr_groups = {}   

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        
        if has_NR:            
            nr_groups.setdefault(plan, []).append(record)
        else:                        
            non_nr_groups.setdefault(plan, []).append(record)

    nr_train_all = []
    nr_tc_all = []
    for plan, records in nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio=0.5)
        nr_train_all.extend(train_records)
        nr_tc_all.extend(tc_records)        

    non_nr_train_all = []
    non_nr_tc_all = []
    for plan, records in non_nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio=0.5)
        non_nr_train_all.extend(train_records)
        non_nr_tc_all.extend(tc_records)
    
    save_jsonl(f"datasets/NR_train_{out_file}.jsonl", nr_train_all)
    save_jsonl(f"datasets/NR_tc_{out_file}.jsonl", nr_tc_all)
    save_jsonl(f"datasets/nonNR_train_{out_file}.jsonl", non_nr_train_all)
    save_jsonl(f"datasets/nonNR_tc_{out_file}.jsonl", non_nr_tc_all)

def integrated_data_spliter(datas, prefix):    
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = {api["plan"] for api in apis}

    datas_by_turn = {}
    for record in datas:
        turn_idx = record.get("turn_idx")
        if turn_idx is None:
            continue
        datas_by_turn.setdefault(turn_idx, []).append(record)

    for turn_idx, new_datas in datas_by_turn.items():
        groups = {}

        for record in new_datas:
            unique_idx = record.get("unique_idx", "")
            parts = unique_idx.split('-')
            if len(parts) < 2:
                continue
            plan = parts[-2]

            N = 5
            others = [k for k in apis_keys if k != plan]
            sampled = random.sample(others, N - 1)
            sampled.append(plan)
            random.shuffle(sampled)
            record["candidates"] = sampled

            groups.setdefault(plan, []).append(record)

        weighted_train_all, weighted_tc_all = [], []
        for plan, records in groups.items():
            random.shuffle(records) 
            weighted_train_records, weighted_tc_records = weighted_split_dataset(records, ratio=0.5, turn_idx=record["turn_idx"])            
            weighted_train_all.extend(weighted_train_records)
            weighted_tc_all.extend(weighted_tc_records)
            print(f"[Turn {turn_idx}] plan '{plan}': {len(train_records)} train, {len(tc_records)} tc")
            print(f"[Turn {turn_idx}] plan '{plan}': {len(weighted_train_records)} weighted train, {len(weighted_tc_records)} weighted tc")

        weighted_train_path = f"datasets/{turn_idx}_weighted_train.tsv"
        weighted_tc_path = f"datasets/{turn_idx}_weighted_tc.tsv"
        pd.DataFrame(weighted_train_all).to_csv(weighted_train_path, sep="\t", index=False)
        pd.DataFrame(weighted_tc_all).to_csv(weighted_tc_path, sep="\t", index=False)
        print(f"[Turn {turn_idx}] saved train: {len(weighted_train_all)} rows -> {weighted_train_path}")
        print(f"[Turn {turn_idx}] saved tc:    {len(weighted_tc_all)} rows -> {weighted_tc_path}")

def data_spliter_pd(datas, prefix, ratio):    
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = set(api["plan"] for api in apis)
    
    nr_groups = {}
    non_nr_groups = {}

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        if has_NR:
            nr_groups.setdefault(plan, []).append(record)
        else:
            if record["rewrited_query"] == record["query"]:                
                continue
            non_nr_groups.setdefault(plan, []).append(record)

    total_plan = set()
    nr_train_all = []
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio)
        nr_train_all.extend(train_records)
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1

    non_nr_train_all = []
    non_nr_tc_all = []
    non_nr_plan_cnt = 0
    for plan, records in non_nr_groups.items():
        train_records, tc_records = split_dataset(records, ratio)
        non_nr_train_all.extend(train_records)
        non_nr_tc_all.extend(tc_records)
        total_plan.add(plan)        
        non_nr_plan_cnt += 1
    
    pd.DataFrame(nr_train_all).to_csv(f"datasets/train/{prefix}_NR_train.tsv", sep="\t", index=False)
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/{prefix}_NR_tc.tsv", sep="\t", index=False)
    pd.DataFrame(non_nr_train_all).to_csv(f"datasets/train/{prefix}_nonNR_train.tsv", sep="\t", index=False)
    pd.DataFrame(non_nr_tc_all).to_csv(f"datasets/tc/{prefix}_nonNR_tc.tsv", sep="\t", index=False)

def data_convert_to_tc(datas, prefix):
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = set(api["plan"] for api in apis)
    
    nr_groups = {}      
    non_nr_groups = {}  

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue  
        plan = parts[-2]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        has_NR = 'NR' in parts[-1]
        if has_NR:
            nr_groups.setdefault(plan, []).append(record)
        else:
            if record["rewrited_query"] == record["query"]:                
                continue
            non_nr_groups.setdefault(plan, []).append(record)

    total_plan = set()    
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        tc_records = records        
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1        

    non_nr_tc_all = []
    non_nr_plan_cnt = 0
    for plan, records in non_nr_groups.items():
        tc_records = records
        non_nr_tc_all.extend(tc_records)
        total_plan.add(plan)        
        non_nr_plan_cnt += 1
    
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/{prefix}_NR_tc.tsv", sep="\t", index=False)    
    pd.DataFrame(non_nr_tc_all).to_csv(f"datasets/tc/{prefix}_nonNR_tc.tsv", sep="\t", index=False)  

def droid_convert_to_tc(datas, prefix):
    apis = read_jsonl("apis/droidcall_apis.jsonl")
    apis_keys = set(api["name"] for api in apis)
    
    nr_groups = {}       
    non_nr_groups = {}   

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue 
        plan = parts[0]        

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys
        nr_groups.setdefault(plan, []).append(record)

    total_plan = set()    
    nr_tc_all = []
    nr_plan_cnt = 0
    for plan, records in nr_groups.items():
        tc_records = records        
        nr_tc_all.extend(tc_records)
        total_plan.add(plan)
        nr_plan_cnt += 1        
    
    pd.DataFrame(nr_tc_all).to_csv(f"datasets/tc/droidcall_tc.tsv", sep="\t", index=False)    

def convert(records, ratio=0.8):    
    if len(records) == 1:
        return [], records    
    train_count = int(len(records) * ratio)    
    train_records = records[:train_count]
    tc_records = records[train_count:]

    if len(train_records) > 50: 
        train_records = train_records[:50]
        tc_records = tc_records[:50]    
        
    return train_records, tc_records

def data_convert_to_tc_by_refered_turn(datas, prefix):    
    apis = read_jsonl("apis/api_v3.0.1.jsonl")
    apis_keys = set(api["plan"] for api in apis)

    all_data_cnt = 0
    target_data_cnt = 0
    ref_groups = {}

    for record in datas:
        unique_idx = record.get("unique_idx", "")
        parts = unique_idx.split('-')
        if len(parts) < 2:
            continue
        plan = parts[-2]
        all_data_cnt += 1

        N = 5
        random_keys = random.sample([k for k in apis_keys if k != plan], N-1)
        random_keys.append(plan)
        random.shuffle(random_keys)
        record["candidates"] = random_keys

        if record["rewrited_query"] == record["query"]:
            continue
        target_data_cnt += 1

        ref = record.get("refered_turn", "NONE")
        ref_groups.setdefault(ref, []).append(record)

    for ref, recs in ref_groups.items():
        if ref is None or ref == "NONE":
            continue
        
        ref_train_all = []
        ref_tc_all    = []

        plan_groups = {}
        for r in recs:
            plan = r["unique_idx"].split('-')[-2]
            plan_groups.setdefault(plan, []).append(r)

        for plan, grp in plan_groups.items():            
            tc_records = grp
            ref_tc_all.extend(tc_records)

        tc_path    = f"datasets/tc/{prefix}_{ref}_tc.tsv"
        pd.DataFrame(ref_tc_all).to_csv(tc_path,    sep="\t", index=False)


def __main__():
    random.seed(42)
    args = get_arg_parse()    
                    
    if args.d: # data split
        target_file = args.t
        prefix = args.o         
        datas = read_jsonl(target_file)        
        
        #droid_convert_to_tc(datas, "") # python3 dataset_spliter.py --d --t datasets/droidcall_gemini.jsonl --o ""
        #data_spliter_pd(datas, prefix, ratio=0.7)
        #data_convert_to_tc(datas, prefix)
        data_convert_to_tc_by_refered_turn(datas, prefix)
        #integrated_data_spliter(datas, prefix)
    else:        
        filenames = args.t_list
        rewrited_query_dup_checker(filenames)

        filenames = args.t_list
        rewrited_query_dup_printer(filenames)

__main__()   
# TC
# data_convert_to_tc_by_refered_turn()
# 1. python3 dataset_spliter.py --d --t datasets/it5_s1_complex_additional_filtered.jsonl --o it5_complex_additional
# 2. python3 dataset_spliter.py --d --t datasets/it4_s1_complex_additional_filtered.jsonl --o it4_complex_additional
# 3. python3 dataset_spliter.py --d --t datasets/it3_s1_complex_additional_filtered.jsonl --o it3_complex_additional
# > history_modified
# 1. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it5_s1_complex_filtered_modified.jsonl --o it5_complex_history
# 2. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it4_s1_complex_filtered_modified.jsonl --o it4_complex_history
# 3. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it3_s1_complex_filtered_modified.jsonl --o it3_complex_history

# data_convert_to_tc()
# 4. python3 dataset_spliter.py --d --t datasets/it5_s1_various_filtered.jsonl --o it5_various
# 5. python3 dataset_spliter.py --d --t datasets/it4_s1_various_filtered.jsonl --o it4_various # (optional)
# 6. python3 dataset_spliter.py --d --t datasets/it3_s1_various_filtered.jsonl --o it3_various # (optional)
# data_spliter_pd()
# 7. python3 dataset_spliter.py --d --t datasets/it2_s1_filtered.jsonl --o it2
# 7. python3 dataset_spliter.py --d --t datasets/it3_s1_filtered.jsonl --o it3
# 7. python3 dataset_spliter.py --d --t datasets/it4_s1_filtered.jsonl --o it4
# 7. python3 dataset_spliter.py --d --t datasets/it5_s1_filtered.jsonl --o it5
# > history_modified
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it2_s1_filtered_modified.jsonl --o it2_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it3_s1_filtered_modified.jsonl --o it3_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it4_s1_filtered_modified.jsonl --o it4_history
# 7. python3 dataset_spliter.py --d --t datasets/modified_jsonl/it5_s1_filtered_modified.jsonl --o it5_history