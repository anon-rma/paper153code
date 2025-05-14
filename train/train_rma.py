import torch
from datasets import load_dataset, Dataset
from typing import List, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from utils.prompt import SFT_RMA_TRAIN_QWEN3, SFT_RMA_TRAIN_LLAMA, SFT_RMA_TRAIN_PHI4
import pdb
import json
import ast
import copy
import os
import glob
import pandas as pd

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "Qwen/Qwen3-1.7B"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "microsoft/Phi-4-mini-instruct"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.gradient_checkpointing_enable()

train_path = '../datasets/train/'

files = [file for file in glob.glob("../datasets/train/rewrited/*.tsv", recursive=True) if "_NR_" not in file and file.endswith(".tsv")]
added_files = [file for file in glob.glob("../datasets/rma_train/rewrited/*.tsv", recursive=True)]
files.append("../datasets/train/it2_NR_train.tsv")
files.extend(added_files)
print(files)
dfs = [pd.read_csv(f, sep="\t", dtype=str) for f in files]
df_all = pd.concat(dfs, ignore_index=True, sort=False)
raw_datasets = Dataset.from_pandas(df_all)

max_length = 1536

def preprocess_example_it(example):
    data = {"conversation_history": example["conversation_history"], "query": example["query"]}
    prompt = prompt_template.format(                
        data=json.dumps(data, ensure_ascii=False, indent=2),
        answer={"rewrited_query": example["rewrited_query"]}
    )
    
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )
    input_ids = tokenized["input_ids"]

    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        model_start = prompt.find("assistant<|end_header_id|>")    
    elif model_name == "microsoft/Phi-4-mini-instruct":
        model_start = prompt.find("<|end|><|assistant|>")
    elif "Qwen/" in model_name:
        model_start = prompt.find("<|im_end|><|im_start|>assistant")
        
    if model_start == -1:
        raise ValueError("Prompt does not contain finish prompt")
    model_token_start = len(
        tokenizer(prompt[:model_start], add_special_tokens=False)["input_ids"]
    )
    labels = [-100] * model_token_start + input_ids[model_token_start:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "strprompt": prompt,        
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }

if model_name == "meta-llama/Llama-3.2-3B-Instruct":
    prompt_template = SFT_RMA_TRAIN_LLAMA
elif model_name == "microsoft/Phi-4-mini-instruct":
    prompt_template = SFT_RMA_TRAIN_PHI4
elif "Qwen/" in model_name:
    prompt_template = SFT_RMA_TRAIN_QWEN3

processed_train = raw_datasets.map(
    preprocess_example_it,     
)

print(processed_train[0]["strprompt"])
max_total_length = 0
for example in processed_train:
    total_length = len(example["input_ids"]) + len(example["labels"])
    if total_length > max_total_length:
        max_total_length = total_length
print(f"input_ids + labels: {max_total_length}")

@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        label_pad_token_id = -100
        max_label_length = max(len(f["labels"]) for f in features)
        for f in features:
            padding_length = max_label_length - len(f["labels"])
            f["labels"] = f["labels"] + [label_pad_token_id] * padding_length
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        return batch

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

if 'Phi-4' in model_name or 'Qwen' in model_name:
    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = 16,
        lora_alpha      = 32,
        lora_dropout    = 0.05,
        bias            = "none",
        target_modules  = "all-linear",
    )
elif 'Llama' in model_name:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=[
        "q_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj", "o_proj"
        ], # 0.38
        lora_dropout=0.1,
        bias="none"
    )
else:    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )

model = get_peft_model(model, lora_config)


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

output_dir = f"{model_name.split('/')[1].split('-')[0]}-rma"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    logging_steps=50,
    logging_dir=f"{output_dir}/logs",
    save_strategy="epoch",
    save_total_limit=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    data_collator=data_collator,
    tokenizer=tokenizer,    
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_dir)