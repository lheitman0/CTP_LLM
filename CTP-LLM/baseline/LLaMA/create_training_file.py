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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

file_name = ""

# PARAMETERS
MAX_SIZE = 30000

# ---------------------------------------------------------------------------- #
#                                   Load Data                                  #
# ---------------------------------------------------------------------------- #
df_trial_design = pd.read_csv("GPT_fine_tuning/data/processed/val_small.csv")
pd.set_option('display.max_columns', None)  # Display all columns
print(df_trial_design.head())

# ---------------------------------------------------------------------------- #
#                               Create JSON file                               #
# ---------------------------------------------------------------------------- #
dataset_data = [
    {
        "instruction": "Predict if this trial will transition to the next phase. Return the answer as the corresponding label: Yes or No.",
        "input": row_dict["Trial_Design"],
        "output": row_dict["Phase_Transition"]
    }
    for row_dict in df_trial_design.to_dict(orient="records")
]
 
print(dataset_data[0])

import json
with open(file_name + '.json', "w") as f:
   json.dump(dataset_data, f)
print(f'Succesfully saved: {file_name}')

# ---------------------------------------------------------------------------- #
#                             Create LLaMA dataset                             #
# ---------------------------------------------------------------------------- #
# Each entry has to have this format:
'''
<s>
    [INST] 
        <<SYS>>
            Predict if this trial will transition to the next phase. Return the answer as the corresponding label: Yes or No.
        <</SYS>>
        TRIAL NAME: Phase II - CLN-PRO-V004; BRIEF: This study will evaluate 
        how well Humacyte's Human Acellular Vessel (HAV) works when surgically 
        implanted into a leg to improve blood flow in patients with peripheral 
        arterial disease (PAD). This study will also evaluate how safe it is to 
        use the HAV in this manner. ; DRUG USED: Humacyl"
    [/INST]
    No
</s>

'''

# Read the JSON file into a DataFrame
df_llama = pd.read_json(file_name + '.json')#.sample(n=1200, random_state=42)

# Function to transform each entry into the desired format
def transform_entry(entry):
    instruction = entry['instruction']
    input_text = entry['input']
    output = entry['output']
    return f"<s>[INST]<<SYS>>{instruction}<</SYS>>{input_text}[/INST]{output}</s>"

# Apply the transformation function to each row of the DataFrame
df_llama['text'] = df_llama.apply(transform_entry, axis=1)

# Remove all instances of quotation marks from the text
df_llama = df_llama.apply(lambda x: x.str.replace('"', ''))
df_llama = df_llama.apply(lambda x: x.str.replace("'", ''))

# Create a new DataFrame with the transformed data
llama_instruction_df = pd.DataFrame(df_llama['text'])

# Display the transformed DataFrame
print(llama_instruction_df.head())
llama_instruction_df.to_csv(file_name + '_llama.csv', index=False, encoding='utf-8')
llama_instruction_df.to_parquet(file_name + '_llama.parquet', index=False)