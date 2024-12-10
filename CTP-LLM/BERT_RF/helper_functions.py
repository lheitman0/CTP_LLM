import pandas as pd
import numpy as np

# Define a function to convert string representations to arrays
def convert(x):
    try:
        return np.fromstring(x[1:-1], sep=' ')
    except (TypeError, ValueError):
        return x

def convert_to_array(df, columns):
    print('Converting df to array ...')
    # Apply the function to all columns in the DataFrame
    df = df.applymap(convert)

    # Now, you can access the first value and its type in any column
    first_value = df.iloc[0, 0]  # Access the first value in the first column
    value_type = type(first_value)

    print(first_value)
    print(value_type)
    print(first_value.shape[0])
    print(df.shape)
    print(df.head())

    # Visualize shape
    shape_df = df.apply(lambda col: col.apply(lambda x: np.shape(x)))
    print(shape_df)

    for col in columns:
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
   
    return df