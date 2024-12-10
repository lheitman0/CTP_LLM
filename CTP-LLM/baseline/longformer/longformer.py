from huggingface_hub import login
import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import os

from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

# Load data
df_train = pd.read_csv('GPT_fine_tuning/data/processed/train_small.csv')
df_train = df_train[df_train['Phase_Transition'] != 'Unknown']
df_train['Phase_Transition'] = df_train['Phase_Transition'].replace('Yes', 1)
df_train['Phase_Transition'] = df_train['Phase_Transition'].replace('No', 0)
df_train.rename(columns={'Phase_Transition': 'label'}, inplace=True)

df_val = pd.read_csv('GPT_fine_tuning/data/processed/val_small.csv')
df_val = df_val[df_val['Phase_Transition'] != 'Unknown']
df_val['Phase_Transition'] = df_val['Phase_Transition'].replace('Yes', 1)
df_val['Phase_Transition'] = df_val['Phase_Transition'].replace('No', 0)
df_val.rename(columns={'Phase_Transition': 'label'}, inplace=True)

# Balance data

from datasets import Dataset
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
#ds = ds_pandas.train_test_split(test_size=0.2)

def preprocess_function(examples): 
	return tokenizer(examples["Trial_Design"], truncation=True) 


train_tokenized = ds_train.map(preprocess_function, batched=True) 
test_tokenized = ds_val.map(preprocess_function, batched=True) 
#print(train_tokenized[0])


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 

id2label = {0: "NEGATIVE", 1: "POSITIVE"} 
label2id = {"NEGATIVE": 0, "POSITIVE": 1} 

model = AutoModelForSequenceClassification.from_pretrained( 
	"allenai/longformer-base-4096", 
num_labels=2, 
id2label=id2label, 
label2id=label2id 
) 

import numpy as np 
import evaluate 
from transformers import DataCollatorWithPadding 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

accuracy = evaluate.load("accuracy") 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 

training_args = TrainingArguments( 
	output_dir="long_sequence_classification", 
	learning_rate=2e-5, 
	per_device_train_batch_size=2, 
	per_device_eval_batch_size=2, 
	num_train_epochs=10, 
	weight_decay=0.01, 
	evaluation_strategy="epoch", 
	save_strategy="epoch", 
	load_best_model_at_end=True, 
	# push_to_hub=True, 
) 

trainer = Trainer( 
	model=model, 
	args=training_args, 
	train_dataset=train_tokenized, 
	eval_dataset=test_tokenized, 
	tokenizer=tokenizer, 
	data_collator=data_collator, 
	compute_metrics=compute_metrics, 
) 

trainer.train() 

OUTPUT_DIR = "results/experiments"
model.save_pretrained(OUTPUT_DIR)


