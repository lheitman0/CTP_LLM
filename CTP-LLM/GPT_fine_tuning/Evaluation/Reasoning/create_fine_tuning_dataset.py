from openai import OpenAI
import pandas as pd
import json
from sklearn.model_selection import train_test_split

#FILE_PATH = 'data/processed/CT_all_phases_small.csv'
FILE_PATH = 'GPT_fine_tuning/data/raw/reasoning_balanced.csv'
MEDICAL_SYSTEM_PROMPT = 'You are a medical expert who specializes in analyzing clinical trials. Your role is to help the user predict whether a clinical trial will transition to the next phase. Answer only with Yes if it transitions to the next phase or No if it doesn\'t. If it does not transition also give an explanation why.'

def create_dataset(question, answer):
    return {
        "messages": [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def create_answer(label, why_stopped):
    if label == 'Yes':
        return label
    else:
        answer = str(label) + ". " + str(why_stopped)
        return answer

def create_json_file(df, file_name):
    with open(file_name, "w") as f:
        for _, row in df.iterrows():
            #example_str = json.dumps(create_dataset(row["Trial_Design"] + DESCRIPTION_EXTENSION, row["Phase_Transition"]))
            example_str = json.dumps(create_dataset(row["Trial_Design"], create_answer(row["Phase_Transition"], row["why_stopped"])))
            f.write(example_str + "\n")
    print('File successfully created as: ', file_name)

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    print('Creating training and validation files ...')

    # Split 
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    create_json_file(df=df_train, file_name="GPT_fine_tuning/data/processed/train_reasoning.jsonl")
    create_json_file(df=df_val, file_name="GPT_fine_tuning/data/processed/val_reasoning.jsonl")

    # Save as csv
    df_train.to_csv("GPT_fine_tuning/data/processed/train_reasoning.csv", index=False)
    df_val.to_csv("GPT_fine_tuning/data/processed/val_reasoning.csv", index=False)
    
