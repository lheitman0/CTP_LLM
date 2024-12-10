from huggingface_hub import login

import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from typing import List
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import torch
from datasets import load_dataset
import pandas as pd
import seaborn as sns
from pylab import rcParams
 
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)

# Check if CUDA is available
if torch.cuda.is_available():
    # Specify the index of the GPU you want to use (assuming you want to use the second GPU)
    gpu_index = 0  # Indexing starts from 0, so 1 corresponds to the second GPU
    # Set the device to the specified GPU
    DEVICE = torch.device(f"cuda:{gpu_index}")
    # Set the device for all future tensors and models
    #torch.cuda.set_device(DEVICE)
else:
    DEVICE = "cpu"

print("Selected device:", DEVICE)

# ---------------------------------------------------------------------------- #
#                                 Model Weights                                #
# ---------------------------------------------------------------------------- #
# Normal LLaMA still requires license
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
 
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True, # loads the model using 8-bit quantization to reduce memory usage and improve inference speed
    torch_dtype=torch.float16,
    device_map="auto",
)
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
#model.to(DEVICE)
 
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
 
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

# ---------------------------------------------------------------------------- #
#                               Load JSON Dataset                              #
# ---------------------------------------------------------------------------- #
data = load_dataset("json", data_files="")
#data = load_dataset("json", data_files="processed/alpaca-bitcoin-sentiment-dataset.json")
print(data["train"])

# ---------------------------------------------------------------------------- #
#                                Create prompts                                #
# ---------------------------------------------------------------------------- #
def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
 
CUTOFF_LEN = 4090

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
 
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


train_val = data["train"].train_test_split(test_size=400, shuffle=True, seed=42)
train_val["train"].to_csv("", index=False)
train_data = (train_val["train"].map(generate_and_tokenize_prompt))

val_test = train_val["test"].train_test_split(test_size=0.5, shuffle=False)
val_test["train"].to_csv("", index=False)
val_test["test"].to_csv("", index=False)
val_data = (val_test["train"].map(generate_and_tokenize_prompt))
test_data = (val_test["test"].map(generate_and_tokenize_prompt))

# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"

# prepare
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
#model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
#underlying_model = model
model.print_trainable_parameters()

# huggingface trainer class
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    eval_accumulation_steps = 20,
    #remove_unused_columns=False
)

# creates batches of input/output sequences for sequence-to-sequence (seq2seq) models
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

import evaluate
import numpy as np
def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = metric.compute(predictions=predictions, references=labels, average='macro')["f1"]
    return {"f1": f1}

# Start training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
 
#model = torch.compile(model)
 
trainer.train()
model.save_pretrained(OUTPUT_DIR)

model.merge_and_unload()