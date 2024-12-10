from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             classification_report,
                             confusion_matrix)

MODEL = 'ft:gpt-4o-mini-2024-07-18:personal::AImoy0yJ'
API_KEY = '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'
FILE_PATH = 'Data/gpt4_test/test_I.csv'
ROLE = 'You are a medical expert who specializes in analyzing clinical trials. Your role is to help the user predict whether a clinical trial will progress to the next phase. Answer only with Yes if it progresses to the next phase or No if it doesn\'t.'

client = OpenAI(
   api_key=API_KEY
)

def evaluate(y_true, y_pred):
    labels = ['Yes', 'No']
    mapping = {'Yes': 1, 'No': 0}
    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Calculate F1-Score
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print(f'F1 Score: {f1:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

def call_API(content):
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": ROLE},
        {"role": "user", "content": content}
        ]
    )

    prediction = completion.choices[0].message.content
    print(prediction)
    return prediction


if __name__ == "__main__":
    # Load the test set into a DataFrame
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    #df = df.iloc[:5]
    print(df)

    predictions = []

    print('Starting prediction ...')
    for index, row in df.iterrows():
        trial = row['Trial_Design']
        prediction = call_API(content=trial)
        
        # Add the prediction to the 'Prediction' column
        predictions.append(prediction)

    df['Prediction'] = predictions
    print(df)
    df.to_csv(f"Data/gpt4_test/pred_PT_per_phase_gpt4mini.csv", index=False)

    # Evaluate performance
    evaluate(y_true=df['Phase_Transition'], y_pred=df['Prediction'])
        