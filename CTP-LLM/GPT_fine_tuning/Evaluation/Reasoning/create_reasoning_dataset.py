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

file_name = "GPT_fine_tuning/data/processed/reasoning"

# PARAMETERS
DROP_PHASE_I = False
DROP_PHASE_II = False
DROP_PHASE_III = False
BALANCE_DATA = False
MAX_SIZE = 30000

# ---------------------------------------------------------------------------- #
#                                   Load Data                                  #
# ---------------------------------------------------------------------------- #
df = pd.read_csv("GPT_fine_tuning/data/processed/CT_reasoning.csv")
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
                                           + ';  SECONDARY OUTCOME: ' + (str(row['secondary_outcome_0']) if pd.notna(row['secondary_outcome_0']) else '') 
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

# Add why_stopped
df_trial_design['why_stopped'] = df['why_stopped']

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
