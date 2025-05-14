import pdb
import json
import random
from utils.frequently_used_tools import read_jsonl, get_arg_parse, get_model_name
from utils.prompt import COMPLEX_MULTI_TURN_GENERATION_PROMPT
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import JsonExtractor

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_datas(datas):    
    processed = []

    for record in datas:
        conv_hist = record.get('conversation_history', [])        
        current_turn = len(conv_hist) + 1

        tokens = record.get('unique_idx', '').split('-')
        plan_candidates = tokens[0::2]

        num_eligible = max(0, current_turn - 2)
        previous_plans = plan_candidates[:num_eligible]

        for turn_idx, prev_plan in enumerate(previous_plans):            
            hist = conv_hist[turn_idx]
            if ' -> ' not in hist:                
                continue
            query, device_response = hist.split(' -> ', 2)

            if query.startswith('turn'):
                query = query[8:]
            
            new_record = {}
            new_record['query'] = query.strip()
            new_record['device_response'] = device_response.strip()
            new_record['next_turn_plan'] = prev_plan
            c_uidx = record['unique_idx']             
            new_record['unique_idx'] = '-'.join(c_uidx.split('-')[:-2])
            new_record['refered_turn'] = turn_idx + 1              
            processed.append(new_record)                

    return processed

def read_all_apis(api_file):
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                api_data = json.loads(line)                
                api_dict[api_data["plan"]] = api_data
    
    return api_dict
    
if __name__ == "__main__":    
    args = get_arg_parse()   
    datas = read_jsonl(args.t)
    apis = read_all_apis(args.api)
    output_path = args.o
    target_datas = process_datas(datas)
    grouped = {}
    for rec in target_datas:
        plan = rec.get('next_turn_plan')

        if plan not in grouped:
            grouped[plan] = []
        grouped[plan].append(rec)

    deduped = {}
    for key, items in grouped.items():
        seen = set()
        deduped[key] = []
        for item in items:
            uid = item['unique_idx']
            if uid not in seen:
                seen.add(uid)
                deduped[key].append(item)    
    
    prompts = []
    prompt_template = COMPLEX_MULTI_TURN_GENERATION_PROMPT
    target_plan_cnt = {}
    for api_key, api_item in apis.items():
        plan_name = api_item["plan"]
        if api_item["arguments"] == {}:            
            continue
        filtered = []
        
        for idx, resp in enumerate(deduped[plan_name]):                   
            try:                
                filtered.append(resp)
            except KeyError:
                continue

        if not filtered:
            continue

        random.seed(153)
        random.shuffle(filtered)
        for group in chunks(filtered, 5):            
            for nt in api_item.get("next_turn_plans", []):
                next_plan = nt["plan"]
                if next_plan not in apis:
                    raise Exception(f"{next_plan}: not found in apis")

                plan_data = apis[next_plan].copy()
                plan_data["reason"] = nt.get("reason")
                plan_data.pop("next_turn_plans", None)
                description = json.dumps(plan_data, indent=2, ensure_ascii=False)

                example = nt["example"].copy()
                example.pop("next_turn_plan", None)
                example["next_turn_plan"] = next_plan
                example = json.dumps(example, indent=2, ensure_ascii=False)
                
                if target_plan_cnt.get(next_plan, 0) > 100:
                    #print(f"Plan `{next_plan}`: {target_plan_cnt[next_plan]} records")
                    continue
                target_plan_cnt[next_plan] = target_plan_cnt.get(next_plan, 0) + 1

                for g in group:                    
                    g["next_turn_plan"] = next_plan

                prev_data = json.dumps(group, indent=2, ensure_ascii=False)
                prompt = prompt_template.format(
                    previous_turn_data=prev_data,
                    description=description,
                    example=example
                )
                prompts.append(prompt)                            
                
    filters = [JsonExtractor()]
    
    model_name, generate_response = get_model_name(args.model)

    def process_prompt(p):
        resp = generate_response("", [p])
        out = []
        for flt in filters:
            out.extend(flt.filter(resp))
        return out

    with open(output_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {executor.submit(process_prompt, p): p for p in prompts}
            for future in tqdm(as_completed(futures), total=len(prompts)):
                try:
                    results = future.result()
                    for item in results:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("Error processing prompt:", e)
    
