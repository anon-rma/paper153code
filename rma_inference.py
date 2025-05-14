import json
import re
import os
import requests
import ast
from datasets import load_dataset
from utils.frequently_used_tools import get_arg_parse, read_jsonl, get_model_name
import pdb
import random
import pandas as pd
from tqdm import tqdm
from filter import JsonExtractor
from functools import partial
from utils.prompt import SFT_RMA_INFERENCE_LLAMA, ZERO_REWRITE_INFERENCE_LLAMA, ZERO_HISTORY_INFERENCE_LLAMA, SFT_REWRITE_INFERENCE_LLAMA, SFT_HISTORY_INFERENCE_LLAMA
from utils.prompt import SFT_RMA_INFERENCE_PHI4, SFT_RMA_INFERENCE_QWEN3

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

def generate_text(prompt, model='llama3-3b-it:latest', host='http://localhost:11434'):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "format": "json",
                "num_predict": 200,               
                "stop": ["}"]
            },
            "stream": False
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        raise Exception(f"API request failed: {response.text}")

def parse_response_json(text):
    text = re.sub(r".*?```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Valid JSON not found")
    return json.loads(match.group())

def fix_unescaped_quotes(raw_str):    
    def repl(match):
        content = match.group(0)
        fixed = content.replace("'", "\\'")
        return fixed
    
    fixed_raw = re.sub(r'"[^"]*"', repl, raw_str)
    return fixed_raw

def extract_json_from_markdown(text):    
    try:        
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:            
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("Valid JSON not found")

        return json.loads(json_str)

    except Exception as e:
        print(e)
        return None

def get_examples(datas):    
    random.shuffle(datas)
    print(len(datas))

    new_datas = []
    for data in datas:
        data = {
            "conversation_history": data["conversation_history"],
            "query": data["query"],
            "rewrited_query": data["rewrited_query"]
        }

        new_datas.append(data)
    
    return new_datas

def save_with_generated_queries(file_path, results, model_name):   
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    gen_list = [item.get('rewrited_query', '') for item in results]

    if len(gen_list) < len(df):
        gen_list.extend([''] * (len(df) - len(gen_list)))

    df['rewrited_query'] = gen_list
    
    filename = os.path.basename(file_path)  
    filename_no_ext = filename.split('.tsv')[0]    
    out_path = f"datasets/tc/phi-addi_rewrited/{filename_no_ext}.tsv"        
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8-sig')
    print(f"Saved updated dataset with `gen_rewrite_query` to: {out_path}")


def main():        
    rewrited_examples = read_jsonl("examples.jsonl") 
    rewrited_examples = get_examples(rewrited_examples)
    
    model_names = ['llama3-3b-it:latest', 'llama3-3b-rma:latest',
     'phi4-mini-rma:latest', 'Phi-half-rma:latest',
     'phi4-mini-rma:latest', 'Qwen3-half-rma:latest',
     'Llama-half-rma:latest', 'Qwen2.5-half-rma:latest',
     'Qwen3-1.7b-half-rma:latest', 'Qwen3-0.6b-half-rma:latest',
     'Qwen3-4B-rewrite-rma-rewrite-integrated-half:latest',
     'Phi-4-mini-instruct-rewrite-rma-rewrite-integrated-1st:latest',
     'phi-addi-rma:latest', 'Phi-half-rma:latest'] 
    model_name = model_names[-1]    
    print(model_name)

    test_type = 'sft'
    if "Llama" in model_name:
        prompt_template = SFT_RMA_INFERENCE_LLAMA
    elif "phi" in model_name or "Phi" in model_name:
        prompt_template = SFT_RMA_INFERENCE_PHI4
    elif "Qwen3" in model_name or "Qwen2.5" in model_name:
        prompt_template = SFT_RMA_INFERENCE_QWEN3    
    
    data_files = {
        'base': [            
            'datasets/tc/it2_NR_tc.tsv',
            'datasets/tc/it2_nonNR_tc.tsv',
            'datasets/tc/it3_nonNR_tc.tsv',
            'datasets/tc/it4_nonNR_tc.tsv',
            'datasets/tc/it5_nonNR_tc.tsv',
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',            
        ], 
        'tmp': [            
            'datasets/train/it2_NR_train.tsv',            
            'datasets/train/it2_nonNR_train.tsv.tsv',
            'datasets/train/it3_nonNR_train.tsv.tsv',
            'datasets/train/it4_nonNR_train.tsv.tsv',
            'datasets/train/it5_nonNR_train.tsv.tsv',
        ],
        'complex': [                        
            'datasets/tc/it3_complex_1_tc.tsv',
            'datasets/tc/it4_complex_1_tc.tsv',
            'datasets/tc/it4_complex_2_tc.tsv',
            'datasets/tc/it5_complex_1_tc.tsv',
            'datasets/tc/it5_complex_2_tc.tsv',
            'datasets/tc/it5_complex_3_tc.tsv',                    
        ],
        'difficult': [                        
            'datasets/tc/it5_various_nonNR_tc.tsv',
            'datasets/tc/it5_various_dNR_tc.tsv',            
        ],
    }        
    tc_type = 'base'  # 'base', 'complex', 'difficult'
    
    def preprocess_example_it(example, prompt_template, test_type, rewrited_examples=None):                
        if test_type == "sft":                        
            data = {"conversation_history": example["conversation_history"], "query": example["query"]}
            prompt = prompt_template.format(                
                data=json.dumps(data, ensure_ascii=False, indent=2),            
            )
        elif "few" in test_type:
            f_ex = ""
            for data in rewrited_examples:                                
                f_ex += json.dumps(data, ensure_ascii=False, indent=2) + "\n"

            data = {"conversation_history": example["conversation_history"], "query": example["query"]}
            prompt = prompt_template.format(
                data=json.dumps(data, ensure_ascii=False, indent=2),
                examples=f_ex,
            )
        return {
            "strprompt":    prompt,
            "stranswer":    example["rewrited_query"],                        
            "query":        example["query"],
            "conversation_history": example["conversation_history"],
        }

    
    all_results = []
    print(data_files[tc_type])
    for file_path in data_files[tc_type]:
        ds = load_dataset('csv', data_files={'tc':[file_path]}, delimiter='\t')['tc']        
                 
        proc = ds.map(
            partial(preprocess_example_it, prompt_template=prompt_template, test_type=test_type, rewrited_examples=rewrited_examples)
        )
        
        print(proc[0]["strprompt"])        
        
        file_results = []

        for ex in tqdm(proc, desc=f"Processing {os.path.basename(file_path)}"):
            prompt = ex["strprompt"]                          
            try:                               
                raw = generate_text(prompt, model=model_name)                                 
                raw = raw + '}'                
                print('--'*20)
                print({"rewrited_query": ex.get("rewrited_query")})
                print(raw)
            except Exception as e:                
                print(f"Error: {e}")                
            
            try:     
                file_results.append({
                "conversation_history": ex.get("conversation_history"),
                "query":                ex.get("query"),
                "rewrited_query":       ast.literal_eval(raw)["rewrited_query"],                                
                "gt":                   ex["stranswer"],                
                })
            except Exception as e:                          
                print(raw.split("rewrited_query")[1].split(': ')[1].split('}')[0][1:])                
                file_results.append({
                    "conversation_history": ex.get("conversation_history"),
                    "query":                ex.get("query"),
                    "rewrited_query":       raw.split("rewrited_query")[1].split(': ')[1].split('}')[0][1:],                                
                    "gt":                   ex["stranswer"],                
                })            
            

        df_file = pd.DataFrame(file_results)        
        all_results.extend(file_results)
        save_with_generated_queries(file_path, file_results, model_name)
        
if __name__ == "__main__":    
    main()
