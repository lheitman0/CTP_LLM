import pandas as pd

# Load the CSV file into a DataFrame
name = 'pred_test_PT_date'
df = pd.read_csv(f'GPT_fine_tuning/data/date_accurate/predictions/{name}.csv')

# Define the identifiers
identifiers = ['TRIAL NAME:', 'BRIEF:', 'DRUG USED:', 'DRUG CLASS:', 'INDICATION:', 'TARGET:', 'THERAPY:', 'LEAD SPONSOR:', 'CRITERIA:', 'PRIMARY OUTCOME:', 'SECONDARY OUTCOME 1:']

# Initialize lists to store data for each part
trial_name = []
brief = []
drug_used = []
drug_class = []
indication = []
target = []
therapy = []
lead_sponsor = []
criteria = []
primary_outcome = []
secondary_outcome_1 = []

# Iterate over each entry in the DataFrame
for entry in df['Trial_Design']:
    # Initialize variables to store data for the current entry
    current_trial_name = ''
    current_brief = ''
    current_drug_used = ''
    current_drug_class = ''
    current_indication = ''
    current_target = ''
    current_therapy = ''
    current_lead_sponsor = ''
    current_criteria = ''
    current_primary_outcome = ''
    current_secondary_outcome_1 = ''
    
    # Split the entry into different parts based on the identifiers
    parts = entry.split(';')
    for part in parts:
        # Check which identifier the part belongs to and assign it to the corresponding variable
        if 'TRIAL NAME:' in part:
            current_trial_name = part.replace('TRIAL NAME:', '').strip()
        elif 'BRIEF:' in part:
            current_brief = part.replace('BRIEF:', '').strip()
        elif 'DRUG USED:' in part:
            current_drug_used = part.replace('DRUG USED:', '').strip()
        elif 'DRUG CLASS:' in part:
            current_drug_class = part.replace('DRUG CLASS:', '').strip()
        elif 'INDICATION:' in part:
            current_indication = part.replace('INDICATION:', '').strip()
        elif 'TARGET:' in part:
            current_target = part.replace('TARGET:', '').strip()
        elif 'THERAPY:' in part:
            current_therapy = part.replace('THERAPY:', '').strip()
        elif 'LEAD SPONSOR:' in part:
            current_lead_sponsor = part.replace('LEAD SPONSOR:', '').strip()
        elif 'CRITERIA:' in part:
            current_criteria = part.replace('CRITERIA:', '').strip()
        elif 'PRIMARY OUTCOME:' in part:
            current_primary_outcome = part.replace('PRIMARY OUTCOME:', '').strip()
        elif 'SECONDARY OUTCOME 1:' in part:
            current_secondary_outcome_1 = part.replace('SECONDARY OUTCOME 1:', '').strip()
    
    # Append the data for the current entry to the corresponding lists
    trial_name.append(current_trial_name)
    brief.append(current_brief)
    drug_used.append(current_drug_used)
    drug_class.append(current_drug_class)
    indication.append(current_indication)
    target.append(current_target)
    therapy.append(current_therapy)
    lead_sponsor.append(current_lead_sponsor)
    criteria.append(current_criteria)
    primary_outcome.append(current_primary_outcome)
    secondary_outcome_1.append(current_secondary_outcome_1)

# Create a new DataFrame with columns for each part
new_df = pd.DataFrame({
    'TRIAL NAME': trial_name,
    'BRIEF': brief,
    'DRUG USED': drug_used,
    'DRUG CLASS': drug_class,
    'INDICATION': indication,
    'TARGET': target,
    'THERAPY': therapy,
    'LEAD SPONSOR': lead_sponsor,
    'CRITERIA': criteria,
    'PRIMARY OUTCOME': primary_outcome,
    'SECONDARY OUTCOME 1': secondary_outcome_1,
    'label': df['Phase_Transition'],
    'prediction': df['Prediction']
})

# Display the new DataFrame
print(new_df)

new_df.to_csv(f'GPT_fine_tuning/data/date_accurate/predictions/PT_CTP_predictions.csv', index=False, encoding='utf-8')