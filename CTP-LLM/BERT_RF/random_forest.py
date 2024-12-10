# import packages 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score 
np.random.seed(1234)
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', None)

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


FILE_NAME_TRAIN = 'BERT_RF/data/processed/train_explore_I_encoded.csv'
FILE_NAME_VAL = 'BERT_RF/data/processed/val_explore_I_encoded.csv'
FILE_NAME_TEST = 'BERT_RF/data/processed/val_explore_I_encoded.csv'

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
    # make prediction 
    preds = classifier.predict(X) 

    # check performance
    print('\nModel accuracy score: {0:0.4f}\n'. format(accuracy_score(y, preds)))

    # Print model Evaluation
    accuracies = accuracy_score(y, preds)
    recalls = recall_score(y, preds)
    precisions = precision_score(y, preds)
    f1s = f1_score(y, preds)

    data = {'precision': precisions, 'recall': recalls, 'f1': f1s, 'accuracy': accuracies}
    table = pd.DataFrame(data, index=[0])  # Add an index to the DataFrame
    print(table)


if __name__ == "__main__":
    X_train, y_train = prepare_data(file_name=FILE_NAME_TRAIN)
    X_valid, y_valid = prepare_data(file_name=FILE_NAME_VAL)
    #X_train, X_valid, y_train, y_valid = train_test_split(X_scaled,y,test_size = 0.2,stratify=y, random_state=1)

    classifier = RandomForestClassifier()
    classifier.fit(X_train,y_train)

    # make prediction 
    print("Validation set:")
    evaluate(classifier=classifier, X=X_valid, y=y_valid)

    X_test, y_test = prepare_data(file_name=FILE_NAME_TEST)
    print("Test set:")
    evaluate(classifier=classifier, X=X_test, y=y_test)

    #X_I, y_I = prepare_data(file_name="baseline/RF/data/processed/CT_test_I.csv")
    #X_II, y_II = prepare_data(file_name="baseline/RF/data/processed/CT_test_II.csv")
    #X_III, y_III = prepare_data(file_name="baseline/RF/data/processed/CT_test_III.csv")

    # make prediction 
    #print("test_I set:")
    #evaluate(classifier=classifier, X=X_I, y=y_I)

    # make prediction 
    #print("test_II set:")
    #evaluate(classifier=classifier, X=X_II, y=y_II)

    # make prediction 
    #print("test_III set:")
    #evaluate(classifier=classifier, X=X_III, y=y_III)

    
