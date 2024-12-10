from openai import OpenAI
import pandas as pd
import json
from sklearn.model_selection import train_test_split

#FILE_PATH = 'data/processed/CT_all_phases_small.csv'
FILE_PATH = 'data/processed/explore_I.csv'
MEDICAL_SYSTEM_PROMPT = 'You are a medical expert who specializes in analyzing clinical trials. Your role is to help the user predict whether a clinical trial will transition to the next phase. Answer only with Yes if it transitions to the next phase or No if it doesn\'t.'
DESCRIPTION_EXTENSION = '\n\nBased on the provided clinical trial description and your extensive background knowledge, do you predict this trial will transition to the next phase?'

def create_dataset(question, answer):
    return {
        "messages": [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def modify_dataframe(df):
    df['Phase_Transition'] = df['Phase_Transition'].replace('Yes', 'Yes. This trial will likely progress to the subsequent phase.')
    df['Phase_Transition'] = df['Phase_Transition'].replace('No', 'No. This trial will likely not progress to the subsequent phase.')
    return df

def create_json_file(df, file_name):
    with open(file_name, "w") as f:
        for _, row in df.iterrows():
            #example_str = json.dumps(create_dataset(row["Trial_Design"] + DESCRIPTION_EXTENSION, row["Phase_Transition"]))
            example_str = json.dumps(create_dataset(row["Trial_Design"], row["Phase_Transition"]))
            f.write(example_str + "\n")
    print('File successfully created as: ', file_name)

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    print('Creating training and validation files ...')
    #df = modify_dataframe(df)

    # Split 
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    create_json_file(df=df_train, file_name="GPT_fine_tuning/data/processed/train_explore_I.jsonl")
    create_json_file(df=df_val, file_name="GPT_fine_tuning/data/processed/val_explore_I.jsonl")

    # Save as csv
    df_train.to_csv("GPT_fine_tuning/data/processed/train_explore_I.csv", index=False)
    df_val.to_csv("GPT_fine_tuning/data/processed/val_explore_I.csv", index=False)
    
