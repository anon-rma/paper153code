import pdb
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import JsonExtractor
from utils.frequently_used_tools import get_model_name, get_arg_parse
from utils.prompt import SIMPLE_MULTI_TURN_GENERATION_PROMPT, DIFFICULT_MULTI_TURN_GENERATION_PROMPT

def chunks(lst, n):
    """리스트 lst를 n개씩 나눠 반환"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    # 1. 파싱 및 파일 설정
    args = get_arg_parse()
    target_file = args.s       # e.g. "datagen/it1_s2_gemini.jsonl"
    api_file    = args.api     # e.g. "apis/api_v3.0.1.jsonl"
    output_path = args.o       # e.g. "datagen/it1_s3_gemini.jsonl"

    # 2. 타겟 데이터 로드
    target_datas = []
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                target_datas.append(json.loads(line))

    # 3. API 사전 로드
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                api_dict[entry["plan"]] = entry

    # 4. prompts 생성
    prompts = []
    prompt_template = SIMPLE_MULTI_TURN_GENERATION_PROMPT

    for api_key, api_item in api_dict.items():
        func_name = api_item["plan"]
        # answer.plan 이 func_name 인 항목만 필터
        filtered = []
        for resp in target_datas:
            try:
                if resp["answer"]["plan"] == func_name and "device_response" in resp:
                    filtered.append({
                        "conversation_history": resp["conversation_history"],
                        "query": resp["query"],
                        "device_response": resp["device_response"]
                    })
            except KeyError:
                continue
        if not filtered:
            continue

        # 10개씩 묶어서 next_turn_plans 처리
        for group in chunks(filtered, 5):
            for nt in api_item.get("next_turn_plans", []):
                next_plan = nt["plan"]
                if next_plan not in api_dict:
                    raise Exception(f"{next_plan}: not found in api_dict")

                # 예제 및 설명 JSON화
                func_data = api_dict[next_plan].copy()
                func_data["reason"] = nt.get("reason")
                func_data.pop("next_turn_plans", None)
                description = json.dumps(func_data, indent=2, ensure_ascii=False)

                example = nt["example"].copy()
                example.pop("next_turn_plan", None)
                example["next_turn_plan"] = next_plan
                example = json.dumps(example, indent=2, ensure_ascii=False)

                for g in group:
                    g["next_turn_plan"] = next_plan

                prev_data = json.dumps(group, indent=2, ensure_ascii=False)
                prompt = prompt_template.format(
                    previous_turn_data=prev_data,
                    description=description,
                    example=example
                )
                prompts.append(prompt)
            #break # 테스트

    # 5. 모델 및 필터 초기화
    filters = [JsonExtractor()]
    model_name, generate_response = get_model_name(args.model)

    # 6. 병렬 처리 함수 정의
    def process_prompt(p):
        resp = generate_response("", [p])
        out = []
        for flt in filters:
            out.extend(flt.filter(resp))
        return out

    # 7. ThreadPoolExecutor 로 동시 API 호출 및 결과 저장
    with open(output_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_prompt, p): p for p in prompts}
            for future in tqdm(as_completed(futures), total=len(prompts)):
                try:
                    results = future.result()
                    for item in results:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("Error processing prompt:", e)

if __name__ == "__main__":
    main()
