from openai import OpenAI
import pandas as pd
from similarity_measure import filter_duplicate_tasks
import re

TASKS_FILE = 'data/processed/tasks_5.csv'

client = OpenAI(
   api_key='api_key',
)

# Load trial data
df_labled = pd.read_csv('data/processed/labled_trial_text.csv')
random_trial = df_labled['text'].sample(n=1).iloc[0]
trial_info = str(random_trial)
#trial_info = ''
print(trial_info)

# Load seed tasks file
df_seed = pd.read_excel('data/raw/seed.xlsx')
#print(df_seed.head())

# Load generated tasks file
try:
    df_generated_tasks = pd.read_csv(TASKS_FILE)
except FileNotFoundError:
    # Create an empty DataFrame if the file doesn't exist
    df_generated_tasks = pd.DataFrame()

    # Save the empty DataFrame to a new CSV file
    df_generated_tasks.to_csv(TASKS_FILE, index=False)

# Create df for generation of new tasks
df_newly_generated = pd.DataFrame()

# Load 6 random seed questions
random_tasks_seed = df_seed[df_seed['category'] == 3]['tasks'].sample(n=6)

# Check if generated questions already exist
# Combine Seed questions with generated questions
if not df_generated_tasks.empty:
  random_tasks_generated = df_generated_tasks['tasks'].sample(n=2)
  random_tasks = pd.concat([random_tasks_seed, random_tasks_generated], ignore_index=True)
else:
  random_tasks = random_tasks_seed
print(random_tasks)

tasks = ""
# Iterate over the random entries and format them
for i, tasks in enumerate(random_tasks):
    tasks += f" Task {i+1}: {tasks}\n"

role = "You are a medical expert with a focus on creating new creative questions, while also providing the answers.\n\n"
instruction_1 = "You are given a series of questions and the description of a clinical trial. Come up with additional questions that are similar to the provided examples and answer them by using infromation given in the trial description.\n\n"
instruction_3 = "You are given a series of tasks. Come up with additional taks that deal with text generation and modification in the field of clinical trials."

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  temperature=1.2,
  messages=[
    {"role": "system", "content": role},
    {"role": "user", "content": instruction_3 + trial_info + tasks + "\n Q9: ; A9: \nQ10: ; A10: \nQ 11: ; A11: \nQ 12: ; A12:"}# \n Task 12: \n Task 13: \n Task 14:"}
  ]
)

print(completion.choices[0].message)

# Extracting the content from the ChatCompletionMessage
message_content = completion.choices[0].message.content

from typing import List

def separate_entries(content: str) -> List[str]:
    # Split the content into individual entries based on '\n\n'
    entries = content.split('\n')
    print('Entries stripped:')
    print(entries)

    # Separate entries starting with 'Q' and 'A'
    q_entries = [entry for entry in entries if entry.startswith('Q')]
    a_entries = [entry for entry in entries if entry.startswith('A')]

    return q_entries, a_entries

# Separate Q and A entries
q_entries, a_entries = separate_entries(message_content)

# Display the separated entries
print("Q Entries:")
print(q_entries)
print("\nA Entries:")
print(a_entries)

# Removing the 'Task x:' prefix from each task
# Remove 'Task x:' occurrences using regex for each task
q_entries_cleaned = [re.sub(r'Q\d+: ', '', question) for question in q_entries]
a_entries_cleaned = [re.sub(r'A\d+: ', '', answer) for answer in a_entries]
print("Q Entries Cleaned:")
print(q_entries_cleaned)
print("\nA Entries Cleaned:")
print(a_entries_cleaned)

#Filter duplicates
unique_tasks = [task for task in q_entries_cleaned if filter_duplicate_tasks(df=df_seed, task=task, threshold=0.6) == True]
unique_tasks = [task for task in q_entries_cleaned if filter_duplicate_tasks(df=df_generated_tasks, task=task, threshold=0.6) == True]

print(len(unique_tasks))

df_newly_generated['tasks'] = pd.Series(unique_tasks)
df_newly_generated['answers'] = pd.Series(a_entries_cleaned)
df_newly_generated['trial_data'] = trial_info

df_generated = pd.concat([df_generated_tasks, df_newly_generated], ignore_index=True)
df_generated.to_csv(TASKS_FILE, index=False, encoding='utf-8')



'''# Splitting the content into individual tasks
tasks = message_content.split('\n')
tasks = [task.strip() for task in tasks if task.strip()]

# Removing the 'Task x:' prefix from each task
# Remove 'Task x:' occurrences using regex for each task
tasks_str = [re.sub(r'Q\d+: ', '', task) for task in tasks]
tasks_str = [re.sub(r'A\d+: ', '', task) for task in tasks_str]
print(tasks_str)
print(len(tasks_str))

#Filter duplicates
unique_tasks = [task for task in tasks_str if filter_duplicate_tasks(df=df_seed, task=task, threshold=0.7) == True]
unique_tasks = [task for task in unique_tasks if filter_duplicate_tasks(df=df_generated_tasks, task=task, threshold=0.7) == True]

print(len(unique_tasks))'''