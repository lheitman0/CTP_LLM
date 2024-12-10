from openai import OpenAI
import pandas as pd
import numpy as np

FILE_PATH = "GPT_fine_tuning/data/raw/reasoning_balanced_test.csv"
#MODEL = "gpt-3.5-turbo-0125"
MODEL = "ft:gpt-3.5-turbo-0125:personal:med:94Cubggv"
ROLE = 'You are a medical expert who specializes in analyzing clinical trials. Your role is to help the user predict whether a clinical trial will transition to the next phase. Answer only with Yes if it transitions to the next phase or No if it doesn\'t. If it does not transition also give an explanation why.'

client = OpenAI(
   api_key='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',
)

def get_prediction(content):
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": ROLE},
        {"role": "user", "content": content}
        ]
    )

    prediction = completion.choices[0].message.content
    print("Predicted label : " + prediction)
    return prediction

def ask_question(content, prediction):
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": ROLE},
        {"role": "user", "content": content},
        {"role": "assistant", "content": prediction},
        {"role": "user", "content": "How did you come to this answer? Which factor did influence your reasoning the most? Give a detailed explanation."}
        ]
    )

    answer = completion.choices[0].message.content
    print("Reasoning:\n" + answer)
    return answer

if __name__ == "__main__":
    # Load the test set into a DataFrame
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    #df = df.iloc[:5]
    #print(df)

    random_entry = df.sample(n=1)

    print('Starting Q & A ...')
    for index, row in random_entry.iterrows():
        trial = row['Trial_Design']
        print(f'Trial:\n{trial}')
        label = row['Phase_Transition']
        print(f'Label: {label}')
        why_stopped = row['why_stopped']
        print(f'Why stopped?: {why_stopped}')
        prediction = get_prediction(content=trial)
        
        #ask_question(content=trial, prediction=prediction)

    