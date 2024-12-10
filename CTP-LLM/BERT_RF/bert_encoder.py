import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import random

# label encoder for text to float
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

## Hyperparameters ##
FILE_NAME = 'BERT_RF/data/raw/val_explore_I.csv'
SAVE_FOLDER_LOCATION = 'baseline/RF/data/processed/'
BALANCE_DATA = False
MODEL_NAME = 'nlpie/clinical-distilbert-i2b2-2010'
ADDITIONAL_INFO = f'_encoded' # added to name of saved file

#model = SentenceTransformer('nlpie/tiny-biobert')
model = SentenceTransformer(MODEL_NAME)

feature_name_list_bert = ['TRIAL NAME',
    'BRIEF',
    'DRUG USED',
    'DRUG CLASS',
    'INDICATION',
    'TARGET',
    'THERAPY',
    'LEAD SPONSOR',
    'CRITERIA',
    'PRIMARY OUTCOME',
    'SECONDARY OUTCOME 1',]

## Funcions ##
def Count(df, column_name, value):
    result = (df[column_name] == value).sum()
    print(f"Number of rows in df where {column_name} is {value}: {result}")

def BalanceData(df):
    Count(df, 'label', 'Yes')
    Count(df, 'label', 'No')
    pos_val = df.loc[df['label'] == 'Yes']
    neg_val = df.loc[df['label'] == 'No']

    # Find abundant and deficient set
    if len(pos_val) > len(neg_val):
        abundant_set = pos_val
        deficient_set = neg_val
        state_1 = "Passed"
        state_2 = "Failed"
    elif len(pos_val) < len(neg_val):
        abundant_set = neg_val
        deficient_set = pos_val
        state_1 = "Failed"
        state_2 = "Passed"
    else:
        print(f"Data is already balanced.")
        return df

    sampled_abundant_set = abundant_set.sample(n=len(deficient_set), random_state=42)
    df_balanced = pd.concat([sampled_abundant_set, deficient_set])
    df_balanced = df_balanced.sort_index()
    print(f"Balanced dataset to:\n - {state_1} = {len(sampled_abundant_set)}\n - {state_2} = {len(deficient_set)}")
    return df_balanced 

def NormalizeArray(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr

def CreateBertEncoding(df, feature):
    feature_list = df[feature].tolist() 
    # Convert any non-string elements to string
    feature_list = [str(feature) for feature in feature_list]

    embeddings = model.encode(feature_list, show_progress_bar=True)
    print(embeddings[0])
    print(embeddings.shape)
    #print(df.shape)

    # Transform each row into a NumPy array
    embedding_arrays = [np.array(row) for row in embeddings]

    df[feature] = embedding_arrays
    pd.set_option('display.max_columns', None)  # Display all columns
    print(df.head())
    return df

def EncodeFeatures(df, feature_name_list):
    for feature_name in feature_name_list:
        print(feature_name)
        if feature_name == 'label' or feature_name == 'nctid' or feature_name == 'Passed_I' or feature_name == 'Passed_II' or feature_name == 'Passed_III':
            continue
        df = CreateBertEncoding(df, feature_name)
    return df

# Load data
df = pd.read_csv(FILE_NAME)
print(df.columns)
print(df.shape)
print(df.head())

# view the column names of the dataframe
col_names = df.columns.str.strip()
print(col_names)

df['label'] = df['label'].replace({'Yes': 1, 'No': 0})

print(df.shape)

df.replace('#NAME?', pd.NA, inplace=True)  # Replace '#NAME?' with pandas NA
#df.dropna(inplace=True)  # Drop rows with missing values

pd.set_option('display.max_columns', None)  # Display all columns
print(df.head())

# Encode textual labels with BERT
df = EncodeFeatures(df=df, feature_name_list=feature_name_list_bert)

# Drop columns that are not in the list of columns to keep
df = df.drop(columns=df.columns.difference(feature_name_list_bert + ['Last_known_Phase', 'Passed_I', 'Passed_II', 'Passed_III', 'label']))
    
print(f"Shape of training df: {df.shape}")

print(f'Column names:\n{df.columns}')

# Save file

file_name = f'BERT_RF/data/processed/val_explore_I{ADDITIONAL_INFO}.csv'
df.to_csv(file_name, index=False)
print(f'File saved successfully as \"{file_name}\"')
