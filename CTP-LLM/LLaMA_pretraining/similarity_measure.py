# Compute the Rouge-L score between a newly generated tasks and all collected tasks.

import pandas as pd
from rouge_score import rouge_scorer

# Function to calculate ROUGE-L score between two sentences
def calculate_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def filter_duplicate_tasks(df,task,threshold):
    # Calculate ROUGE-L score between the new sentence and existing sentences
    rouge_scores = []
    for sentence in df:
        rouge_scores.append(calculate_rouge_l(sentence, task))
    #print(rouge_scores)

    # Determine if the new sentence is different enough based on a threshold
    is_different_enough = all(score < threshold for score in rouge_scores)
    print("Is the new sentence different enough?", is_different_enough)
    return is_different_enough

# Load data
df_tasks = pd.read_csv('data/processed/tasks.csv')
print(df_tasks.head())

# New sentence to compare
new_sentence = "How would you modify the protocol of the trial in terms of treatment arm?"
filter_duplicate_tasks(df=df_tasks['tasks'],task=new_sentence,threshold=0.7)

