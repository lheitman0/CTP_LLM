import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/Phase_Transition_Dataset.csv")
#column_names = df.columns.tolist()
#print(column_names)

df = df.dropna(subset=['date'])  # Drop rows with NaN in 'date'
df = df[df['date'].str.strip().astype(bool)]

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort the DataFrame by the 'date' column
df = df.sort_values(by='date')
df = df.reset_index(drop=True)

print(df['date'])

# Determine the split index
split_index = int(len(df) * 0.7)

# Split the DataFrame
train_df = df.iloc[:split_index].reset_index(drop=True)
test_df = df.iloc[split_index:].reset_index(drop=True)
train_df.to_csv('data/train_PT.csv', index=False)
test_df.to_csv('data/test_PT.csv', index=False)
