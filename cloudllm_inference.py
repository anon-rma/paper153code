import json
import re
import os
import requests
import ast
from datasets import load_dataset
import pdb
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from utils.prompt import ZERO_SHOT_INFERENCE_GENERATION_PROMPT, ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT
from utils.frequently_used_tools import get_arg_parse, get_model_name
def read_apis(api_file, simple=False):    
    with open(api_file, encoding="utf-8") as f:
        if simple:
            return json.load(f)
        else:
            out = {}
            for line in f:
                data = json.loads(line)
                for k in ("examples","returns","next_turn_plans"):
                    data.pop(k, None)
                out[data["plan"]] = data
            return out

def parse_response_json(text):
    text = re.sub(r".*?```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Valid JSON not found")
    return json.loads(match.group())


def print_eval(df, title=None, test_type=None, detail=False):
    if title: print(f"\n## Performance for {title}\n")
    for col in ("plan","arguments","all"):
        acc = round((df[col]=="pass").mean(), 2)
        print(f"{col.title():<10} Accuracy : {acc}")
    print("-"*40)

    if title is None:
        title = ""
    with open("logs/cloud_inference_log.txt", 'a', encoding='utf-8') as f:
        if title:
            f.write(f"\n## Performance for {title}, {test_type}\n")
        for col in ("plan", "arguments", "all"):
            acc = round((df[col] == "pass").mean(), 2)
            f.write(f"{col.title():<10} Accuracy : {acc}\n")
        f.write("-" * 40 + "\n")

    if detail:
        df["gt_plan"] = df["gt"].apply(lambda x: x.get("plan"))
        detail_df = (
            df.groupby("gt_plan")[["plan","arguments","all"]]
              .apply(lambda x: (x=="pass").mean().round(2))
              .reset_index()
        )
        print(detail_df.to_string(index=False))

def extract_json_from_markdown(text):    
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("json block not found")

        return json.loads(json_str)

    except Exception as e:
        print(e)
        return None

def main(out_file):
    args = get_arg_parse()
    model_name, generate_response = get_model_name(args.model)    

    apis = read_apis("apis/api_v3.0.1.jsonl", simple=False)   
    
    test_type = args.t
    if test_type == 'rewrite':
        prompt_template = ZERO_SHOT_INFERENCE_GENERATION_PROMPT
    elif test_type == 'history':
        prompt_template = ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT    

    tc_type = 'base'
    data_files = {                
        'base': [            
            'datasets/tc/it2_NR_tc.tsv',
            'datasets/tc/it2_nonNR_tc.tsv',
            'datasets/tc/it3_nonNR_tc.tsv',
            'datasets/tc/it4_nonNR_tc.tsv',
            'datasets/tc/it5_nonNR_tc.tsv',            
        ],   
        'complex': [                        
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',                    
        ],                                 
    }        
        
    def preprocess_example_it(example, apis, prompt_template, test_type):
        api_str = ""
        re_fmt  = {"plan": "str type tool", "arguments": {"key1": "value1"}}
        for plan in ast.literal_eval(example["candidates"]):
            api_data = apis[plan].copy()
            api_str += json.dumps(api_data, ensure_ascii=False, indent=2) + "\n"
        
        if test_type == "history":
            prompt = prompt_template.format(
                tools=api_str,
                re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),
                conversation_history=example["conversation_history"],
                data=example["query"]
            )
        elif test_type == "rewrite":
            prompt = prompt_template.format(
                tools=api_str,
                re_format=json.dumps(re_fmt, ensure_ascii=False, indent=2),                
                data=example["rewrited_query"]
            )
        return {
            "strprompt":    prompt,
            "stranswer":    json.dumps(example["answer"], ensure_ascii=False, indent=2),
            "candidates":   example["candidates"],
            "rewrited_query": example["rewrited_query"],
            "query":        example["query"],
            "conversation_history": example["conversation_history"],
        }

    all_results = []
    for file_path in data_files[tc_type]:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']                   
        proc = ds.map(
            partial(preprocess_example_it, apis=apis, prompt_template=prompt_template, test_type=test_type)
        )
        print(proc[0]['strprompt'])        
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]            
            
            try:                
                raw = generate_response("", [prompt])                
                raw = raw[0]["text"]                
                
                try:
                    result = extract_json_from_markdown(raw)                        
                except:                        
                    result = parse_response_json(raw)                    

                gt = ast.literal_eval(ex["stranswer"])
                if type(gt) == str:
                    gt = ast.literal_eval(gt)

                plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                arg_res  = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                all_res  = "pass" if plan_res=="pass" and arg_res=="pass" else "fail"                
            except Exception as e:
                result   = {"error": str(e)}                
                print(f"Error: {e}")
                plan_res = "fail"
                arg_res  = "fail"
                all_res  = "fail"

            file_results.append({
                "conversation_history": ex.get("conversation_history"),
                "query":                ex.get("query"),
                "rewrited_query":       ex.get("rewrited_query"),
                "candidates":           ex.get("candidates"),
                "generation":           result,
                "gt":                   gt,
                "plan":                 plan_res,
                "arguments":            arg_res,
                "all":                  all_res,
                "file":                 os.path.basename(file_path)
            })

        df_file = pd.DataFrame(file_results)
        print_eval(df_file, title=os.path.basename(file_path), test_type=test_type)
        all_results.extend(file_results)

    result = pd.DataFrame(all_results)    
    result.to_csv(out_file, sep='\t', index=False, encoding='utf-8-sig')        
    
if __name__ == "__main__":
    args = get_arg_parse()
    main(args.o)
