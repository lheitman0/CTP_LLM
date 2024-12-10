import pandas as pd
import numpy as np

FILE_PATH = "baseline/RF/data/raw/val_small_split.csv"

def filter_df(df, phase):
    filtered_df = df[df['TRIAL NAME'].str.contains(phase)]
    print(filtered_df)

    # Get indices of rows in filtered_df
    indices_to_drop = filtered_df.index

    # Drop rows from df using the indices
    df = df.drop(indices_to_drop)

    return filtered_df, df

if __name__ == "__main__":
    # Load the test set into a DataFrame
    df = pd.read_csv(FILE_PATH, encoding='utf-8')
    #df = df.iloc[:5]
    print(df)

    # Filter rows containing for phases
    df_III, df = filter_df(df=df, phase='Phase III')
    df_II, df = filter_df(df=df, phase='Phase II')
    df_I, df = filter_df(df=df, phase='Phase I')

    df_III.to_csv("baseline/RF/data/processed/test_III.csv", index=False)
    df_II.to_csv("baseline/RF/data/processed/test_II.csv", index=False)
    df_I.to_csv("baseline/RF/data/processed/test_I.csv", index=False)
    