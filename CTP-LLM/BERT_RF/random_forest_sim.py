# import packages 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score 
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


FILE_NAME_TRAIN = 'Data/BERT_RF/processed/train_explore_I_encoded.csv'
FILE_NAME_VAL = 'Data/BERT_RF/processed/val_explore_I_encoded.csv'
FILE_NAME_TEST = 'Data/BERT_RF/processed/val_explore_I_encoded.csv'

def prepare_data(file_name):
    # Load data
    df = pd.read_csv(file_name)
    print(df.head())

    df_encoded = df[['TRIAL NAME',
        'BRIEF',
        'DRUG USED',
        'DRUG CLASS',
        'INDICATION',
        'TARGET',
        'THERAPY',
        'LEAD SPONSOR',
        'CRITERIA',
        'PRIMARY OUTCOME',
        'SECONDARY OUTCOME 1',]]

    from helper_functions import convert_to_array
    df_array = convert_to_array(df_encoded, df_encoded.columns)
    #print(df_array)

    nan_count = df_array.isna().sum().sum()
    print("Number of df_array NaN entries:", nan_count)

    test = df_array['BRIEF'][0]
    print(type(test))

    # Expand dataframe
    final_dfs = []
    for col in df_array.columns:
        # Create a DataFrame from the list of Series in the column
        final_df = pd.DataFrame([pd.Series(row) for row in df_array[col]])
        # Append the resulting DataFrame to the list of final DataFrames
        final_dfs.append(final_df)

    # Concatenate all final DataFrames along the columns axis
    result_df = pd.concat(final_dfs, axis=1)
    print(result_df.head())
    print(result_df.shape)
    num_columns = len(result_df.columns)
    print("Number of columns:", num_columns)

    nan_count = result_df.isna().sum().sum()
    print("Number of result_df NaN entries:", nan_count)

    # Add labels
    result_df['label'] = df.label.values
    result_df = result_df.fillna(0.0)

    X = result_df.drop(['label'],axis=1)
    y = result_df.label.values 
    X_scaled =  StandardScaler().fit_transform(X) 

    return X_scaled, y

def evaluate(classifier, X, y):
    preds = classifier.predict(X)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds)
    }

def run_multiple_times(n_runs=10):
    all_results = {
        'train': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'valid': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'test': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }

    for _ in range(n_runs):
        X_train, y_train = prepare_data(file_name=FILE_NAME_TRAIN)
        X_valid, y_valid = prepare_data(file_name=FILE_NAME_VAL)
        X_test, y_test = prepare_data(file_name=FILE_NAME_TEST)

        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        train_results = evaluate(classifier, X_train, y_train)
        valid_results = evaluate(classifier, X_valid, y_valid)
        test_results = evaluate(classifier, X_test, y_test)

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            all_results['train'][metric].append(train_results[metric])
            all_results['valid'][metric].append(valid_results[metric])
            all_results['test'][metric].append(test_results[metric])

    return all_results

def print_average_results(results):
    for dataset in ['train', 'valid', 'test']:
        print(f"\nAverage {dataset.capitalize()} set results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            avg_value = np.mean(results[dataset][metric])
            print(f"{metric.capitalize()}: {avg_value:.4f}")

if __name__ == "__main__":
    n_runs = 10
    results = run_multiple_times(n_runs)
    print(f"Results averaged over {n_runs} runs:")
    print_average_results(results)