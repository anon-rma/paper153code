import os, json, ast
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)

#MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
#MODEL_NAME = "Qwen/Qwen3-4B"
#MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_NAME = "Qwen/Qwen3-0.6B"
TOOLS_PATH = "../apis/simple_api.json"
TRAIN_DIR  = "../datasets/train/"
TRAIN_TYPE = "history"                      
PREFIX     = "all_linear"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.pad_token = tokenizer.unk_token

tokenizer.pad_token_id     = tokenizer.eos_token_id
tokenizer.padding_side     = "right"
tokenizer.model_max_length =  1538

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model.gradient_checkpointing_enable()

def read_simple_apis(api_file):
    with open(api_file, encoding="utf-8") as f:
        return json.load(f)

apis = read_simple_apis(TOOLS_PATH)

train_files = [
    os.path.join(TRAIN_DIR, f)
    for f in os.listdir(TRAIN_DIR)
    if ("nonNR" in f and f.endswith(".tsv"))
]
train_files.append(os.path.join(TRAIN_DIR, "it2_NR_train.tsv"))

print(train_files)
raw_ds = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

MAX_LEN =  1536

def preprocess_example(example):        
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:        
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"        

    if TRAIN_TYPE == "history":
        system_msg = f"You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.\n <|tool|>{api_str}<|/tool|>"
        
        user_content = (            
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    elif TRAIN_TYPE == "rewrite":
        system_msg = f"Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.\n <|tool|>{api_str}<|/tool|>"
        
        user_content = (            
            f"User Query: {example['rewrited_query']}"
        )

    assistant_content = (
        json.dumps(example["answer"], ensure_ascii=False)
        if isinstance(example["answer"], dict)
        else example["answer"]
    )

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    prompt_with_answer = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    tok = tokenizer(
        prompt_with_answer,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
    )
    input_ids = tok["input_ids"]

    assistant_start = prompt_with_answer.rfind("<|assistant|>") + len("<|assistant|>")
    label_start = len(
        tokenizer(prompt_with_answer[:assistant_start],
                  add_special_tokens=False)["input_ids"]
    )

    labels = [-100] * label_start + input_ids[label_start:]

    return {
        "input_ids": input_ids,
        "labels":    labels,
        "strprompt": prompt_with_answer,
        "stranswer": assistant_content,
    }

processed_train = raw_ds["train"].map(
    preprocess_example, desc="Applying Qwen chat template"
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
        pad_val = -100
        max_len = max(len(f["labels"]) for f in features)
        for f in features:
            f["labels"] += [pad_val] * (max_len - len(f["labels"]))
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")

data_collator = DataCollatorForCausalLM(tokenizer)

lora_cfg = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    target_modules = "all-linear",
)

model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M")

out_dir = f"{MODEL_NAME.split('/')[-1]}-{TRAIN_TYPE}-{PREFIX}"
training_args = TrainingArguments(
    output_dir                 = out_dir,
    overwrite_output_dir       = True,
    num_train_epochs           = 6,
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 4,
    learning_rate              = 5e-5,
    bf16                       = True,
    gradient_checkpointing     = True,
    max_grad_norm              = 1.0,
    logging_steps              = 50,
    logging_dir                = f"{out_dir}/logs",
    save_strategy              = "epoch",
    save_total_limit           = 8,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = processed_train,
    data_collator = data_collator,
    tokenizer     = tokenizer,
)

trainer.train()

trainer.save_model()
tokenizer.save_pretrained(out_dir)
