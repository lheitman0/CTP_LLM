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

file_name = "data/processed/explore_I"

# PARAMETERS
DROP_PHASE_I = False
DROP_PHASE_II = True
DROP_PHASE_III = True
BALANCE_DATA = True
MAX_SIZE = 30000

# ---------------------------------------------------------------------------- #
#                                   Load Data                                  #
# ---------------------------------------------------------------------------- #
df = pd.read_csv("data/train_PT.csv")
pd.set_option('display.max_columns', None)  # Display all columns
print(df.head())
#print(df.Passed_III.value_counts())

if DROP_PHASE_I:
    print(f'Shape before drop: {df.shape}')
    df = df.loc[df['Last_known_Phase'] != 1]
    print(f'Shape after drop: {df.shape}')

if DROP_PHASE_II:
    print(f'Shape before drop: {df.shape}')
    df = df.loc[df['Last_known_Phase'] != 2]
    print(f'Shape after drop: {df.shape}')

if DROP_PHASE_III:
    print(f'Shape before drop: {df.shape}')
    df = df.loc[df['Last_known_Phase'] != 3]
    print(f'Shape after drop: {df.shape}')

# ---------------------------------------------------------------------------- #
#            Bring data into shape: "Trial_Design" "Phase_Transition"          #
# ---------------------------------------------------------------------------- #
df_trial_design = pd.DataFrame()
# Concatenate texts from 'Column1' and 'Column2' and save them to a new column 'Concatenated'
df_trial_design['Trial_Design'] = df.apply(lambda row: 
                                           'TRIAL NAME: ' + (str(row['Trial_Name']) if pd.notna(row['Trial_Name']) else '') 
                                           + ';  BRIEF: ' + (str(row['brief']) if pd.notna(row['brief']) else '') 
                                           + ';  DRUG USED: ' + (str(row['Name']) if pd.notna(row['Name']) else '') 
                                           + ';  DRUG CLASS: ' + (str(row['Drug_Class']) if pd.notna(row['Drug_Class']) else '') 
                                           + ';  INDICATION: ' + (str(row['Indication']) if pd.notna(row['Indication']) else '') 
                                           + ';  TARGET: ' + (str(row['Target']) if pd.notna(row['Target']) else '') 
                                           + ';  THERAPY: ' + (str(row['Therapy']) if pd.notna(row['Therapy']) else '') 
                                           + ';  LEAD SPONSOR: ' + (str(row['lead_sponsor']) if pd.notna(row['lead_sponsor']) else '') 
                                           + ';  CRITERIA: ' + (str(row['criteria']) if pd.notna(row['criteria']) else '') 
                                           + ';  PRIMARY OUTCOME: ' + (str(row['primary_outcome']) if pd.notna(row['primary_outcome']) else '') 
                                           + ';  SECONDARY OUTCOME 1: ' + (str(row['secondary_outcome_0']) if pd.notna(row['secondary_outcome_0']) else '') 
                                           #+ ';  SECONDARY OUTCOME 2: ' + (str(row['secondary_outcome_1']) if pd.notna(row['secondary_outcome_1']) else '') 
                                           #+ ';  DOSAGE DESCRIPTION: ' + (str(row['Dosage_Description']) if pd.notna(row['Dosage_Description']) else '') 
                                           #+ ';  CONTROL DOSAGE DESCRIPTION: ' + (str(row['Control_Dosage_Description']) if pd.notna(row['Control_Dosage_Description']) else '') 
                                           #+ ';  TRIAL DESCRIPTION: ' + (str(row['description']) if pd.notna(row['description']) else '') 
                                           , axis=1)

df_trial_design = df_trial_design.replace(r'\s+', ' ', regex=True)

# Create label
# Check Last_know_Phase -> choose Passed_X accordingly
# Define a function to map values based on 'Last_known_Phase'
def map_label(row):
    if row['Last_known_Phase'] == 1:
        return row['Passed_I']
    elif row['Last_known_Phase'] == 2:
        return row['Passed_II']
    elif row['Last_known_Phase'] == 3:
        return row['Passed_III']
    else:
        return None  # or any other default value if needed

# Create the new 'Phase_Transition' column based on the values in 'Last_known_Phase'
df_trial_design['Phase_Transition'] = df.apply(map_label, axis=1)
df_trial_design = df_trial_design[df_trial_design['Phase_Transition'] != 'Unknown']
df_trial_design = df_trial_design.dropna(subset=['Phase_Transition'])

# Display the DataFrame with the concatenated texts
pd.set_option('display.max_rows', None)  # Display all rows
print(df_trial_design.head())

pd.set_option('display.max_colwidth', None)
print(df_trial_design['Trial_Design'].iloc[1])
print(df_trial_design.Phase_Transition.value_counts())

if BALANCE_DATA:
    df_yes = df_trial_design[df_trial_design['Phase_Transition'] == 'Yes']
    df_no = df_trial_design[df_trial_design['Phase_Transition'] == 'No']

    if len(df_yes) < len(df_no):
        df_no_sampled = df_no.sample(n=len(df_yes), random_state=42)
        df_balanced = pd.concat([df_no_sampled, df_yes])
    else:
        df_yes_sampled = df_yes.sample(n=len(df_no), random_state=42)
        df_balanced = pd.concat([df_yes_sampled, df_no])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df_trial_design = df_balanced
    print(df_trial_design['Phase_Transition'].value_counts())

dataset_size = len(df_trial_design)
if MAX_SIZE < dataset_size:
    print('Reducing dataset size ...')
    print(f'Initial size: {dataset_size}')
    df_trial_design = df_trial_design.sample(MAX_SIZE, random_state=42)
    print(f'Final size: {len(df_trial_design)}')
    print(df_trial_design['Phase_Transition'].value_counts())

#file_name = 'processed/Trial_Design_II-III'
df_trial_design.to_csv(file_name + '.csv', index=False, encoding='utf-8')

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