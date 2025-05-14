from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root not in sys.path:
    sys.path.insert(0, root)

from utils.frequently_used_tools import get_arg_parse

def merge_model(base_model_path, lora_model_path, merged_model_path):    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    model = model.merge_and_unload()    
    model.save_pretrained(merged_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)


args = get_arg_parse()
if args.model == "llama3":
    base_model_path = ""
elif args.model == "phi4":
    base_model_path = ""
elif args.model == "qwen2.5":
    base_model_path = ""
elif args.model == "qwen3":
    base_model_path = ""
elif args.model == 'qwen3-1.7b':
    base_model_path = ""
elif args.model == 'qwen3-0.6b':
    base_model_path = ""
else:
    print("Invalid model name. Please use 'llama3' or 'phi4'.")
    exit(0)
lora_model_path = args.t1 
merged_model_path = args.t2 
merge_model(base_model_path, lora_model_path, merged_model_path)