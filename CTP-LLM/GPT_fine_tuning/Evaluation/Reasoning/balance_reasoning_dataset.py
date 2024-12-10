import nltk
import pandas as pd
#nltk.download('punkt')  # Download the Punkt tokenizer models if not already downloaded
FILE_PATH = 'GPT_fine_tuning/data/processed/reasoning.csv'

# Function to tokenize text and count tokens
def count_tokens(text):
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    return len(tokens)  # Count the tokens

# Assuming 'text_column' is the name of the column containing the text data
df = pd.read_csv(FILE_PATH, encoding='utf-8')
df['token_count'] = df["Trial_Design"].apply(count_tokens)  # Apply the function to tokenize and count tokens
df = df.sort_values(by='token_count', ascending=False)

# drop long entries
#df = df.drop(df.index[:500])
#df_sorted = df_sorted[df_sorted['token_count'] <= 1200]

# Drop samples where 'Phase_transition' is 'No' and 'why_stopped' is NaN
df = df.drop(df[(df['Phase_Transition'] == 'No') & df['why_stopped'].isna()].index)

df_yes = df[df['Phase_Transition'] == 'Yes']
df_yes_sampled = df_yes.sample(n=2000, random_state=42)
df_no = df[df['Phase_Transition'] == 'No']
df_no_sampled = df_no.sample(n=2500, random_state=42)

df_test = df_no_sampled.iloc[:500]
df_no_sampled = df_no_sampled.iloc[500:]

df = pd.concat([df_no_sampled, df_yes_sampled])
df = df.sort_values(by='token_count', ascending=False)

df_save = df.copy()
df_save.drop('token_count', axis=1, inplace=True)
df_save.to_csv('GPT_fine_tuning/data/raw/reasoning_balanced.csv', index=False)
df_test.drop('token_count', axis=1, inplace=True)
df_test.to_csv('GPT_fine_tuning/data/raw/reasoning_balanced_test.csv', index=False)

# Total number of tokens in the DataFrame
total_tokens = df['token_count'].sum()
print("Total number of tokens in the DataFrame:", total_tokens)

# Count different labels
entries_count = df['Phase_Transition'].value_counts()
print(entries_count)
