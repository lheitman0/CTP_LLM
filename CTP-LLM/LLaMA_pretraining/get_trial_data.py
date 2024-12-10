# This script retrieves trial information from ClinicalTrial.gov
# Its purpose is to fine-tune the LLM on CT related geration tasks,
# before training it on our phase transition prediction task.

import numpy as np
import pandas as pd
import os
import warnings
import random

PREPROCESS_UNLABLED_DATA = False
PREPROCESS_LABLED_DATA = False
GENERATE_TRIAL_DESCRIPTIONS = True
CREATE_QA_DATASET = False
CREATE_LLAMA_DATA = False

TARGET_ENTRIES = 100000

# check xml files for trial outcome
from tqdm import tqdm
from xml.etree import ElementTree as ET
def CheckXML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    nctid = root.find('id_info').find('nct_id').text

    try:
        study_type = root.find('study_type').text 
        if study_type != 'Interventional':
            return None
    except:
        return None

    try:
        title = root.find('official_title').text
        # print("enrollment:", enrollment)
    except:
        try:
            title = root.find('brief_title').text
        except:
            title = ''

    try:
        status = root.find('overall_status').text 
        #print("status:", status)
    except:
        status = ''

    try:
        phase = root.find('phase').text
        #print("phase:", phase)
        if phase == 'N/A':
            #After xml filter: (229, 38)
            phase = ''
    except:
        phase = ''

    try:
        criteria = root.find('eligibility').find('criteria').find('textblock').text
        # print('found criteria')
    except:
        criteria = ''

    try:
        enrollment = root.find('enrollment').text
        # print("enrollment:", enrollment)
    except:
        enrollment = ''

    try:
        lead_sponsor = root.find('sponsors').find('lead_sponsor').find('agency').text 
        # print("lead_sponsor:", lead_sponsor)
    except:
        lead_sponsor = ''

    try:
        brief = root.find('brief_summary').find('textblock').text
    except:
        brief = ''

    try:
        description = root.find('detailed_description').find('textblock').text
    except:
        description = ''

    try:
        primary_outcome_element = root.find('primary_outcome')
        if primary_outcome_element is not None:
            po_measure_element = primary_outcome_element.find('measure')
            po_measure = f"MEASURE: {po_measure_element.text}; " if po_measure_element is not None else ''
            
            po_time_frame_element = primary_outcome_element.find('time_frame')
            po_time_frame = f"TIME_FRAME: {po_time_frame_element.text}; " if po_time_frame_element is not None else ''
            
            po_description_element = primary_outcome_element.find('description')
            po_description = f"DESCRIPTION: {po_description_element.text}; " if po_description_element is not None else ''
            
            if (po_measure != '') | (po_time_frame != '') | (po_description != ''):
                primary_outcome = f"PRIMARY OUTCOME: <{po_measure}{po_time_frame}{po_description}>"
        else:
            primary_outcome = ''
    except:
        primary_outcome = ''

    try:
        secondary_outcomes = root.findall('.//secondary_outcome')

        # Initialize a list to store the extracted secondary outcomes
        secondary_outcome = []

        # Iterate over each secondary_outcome element
        for i, secondary_outcome_entry in enumerate(secondary_outcomes, start=1):
            so_measure_element = secondary_outcome_entry.find('measure')
            so_measure = f"MEASURE: {so_measure_element.text}; " if so_measure_element is not None else ''

            so_time_frame_element = secondary_outcome_entry.find('time_frame')
            so_time_frame = f"TIME_FRAME: {so_time_frame_element.text}; " if so_time_frame_element is not None else ''

            so_description_element = secondary_outcome_entry.find('description')
            so_description = f"DESCRIPTION: {so_description_element.text}; " if so_description_element is not None else ''

            if (so_measure != '') | (so_time_frame != '') | (so_description != ''):
                so_outcome = f"SECONDARY OUTCOME {i}: {so_measure}{so_time_frame}{so_description}>; "
                secondary_outcome.append(so_outcome)

        secondary_outcome = ' '.join(secondary_outcome)
    except:
        secondary_outcome = ''

    try:
        study_design_info = root.find('study_design_info')

        sd_allocation_element = study_design_info.find('allocation')
        sd_allocation = f"ALLOCATION: {sd_allocation_element.text}; " if sd_allocation_element is not None else ''

        sd_intervention_model_element = study_design_info.find('intervention_model')
        sd_intervention_model = f"INTERVENTION MODEL: {sd_intervention_model_element.text}; " if sd_intervention_model_element is not None else ''

        sd_intervention_model_description_element = study_design_info.find('intervention_model_description')
        sd_intervention_model_description = f"DESCRIPTION: {sd_intervention_model_description_element.text}; " if sd_intervention_model_description_element is not None else ''

        sd_primary_purpose_element = study_design_info.find('primary_purpose')
        sd_primary_purpose = f"PRIMARY PURPOSE: {sd_primary_purpose_element.text}; " if sd_primary_purpose_element is not None else ''

        sd_masking_element = study_design_info.find('masking')
        sd_masking = f"MASKING: {sd_masking_element.text}; " if sd_masking_element is not None else ''

        if (sd_allocation != '') | (sd_intervention_model != '') | (sd_intervention_model_description != '') | (sd_primary_purpose != '') | (sd_masking != ''):
            study_design = f"{sd_allocation}{sd_intervention_model}{sd_intervention_model_description}{sd_primary_purpose}{sd_masking}"
        else:
            study_design = ''
    except:
        study_design = ''

    try:
        arm_groups = root.findall('.//arm_group')

        # Initialize a list to store the extracted arm groups
        arm_group = []

        # Iterate over each arm_group element
        for i, arm_group_entry in enumerate(arm_groups, start=1):
            ag_label_element = arm_group_entry.find('arm_group_label')
            ag_label = f"LABEL: {ag_label_element.text}; " if ag_label_element is not None else ''

            ag_group_type_element = arm_group_entry.find('arm_group_type')
            ag_group_type = f"GROUP TYPE: {ag_group_type_element.text}; " if ag_group_type_element is not None else ''

            ag_description_element = arm_group_entry.find('description')
            ag_description = f"DESCRIPTION: {ag_description_element.text}; " if ag_description_element is not None else ''

            if (ag_label != '') | (ag_group_type != '') | (ag_description != ''):
                ag_group = f"ARM GROUP {i}: <{ag_label}{ag_group_type}{ag_description}>; "
                arm_group.append(ag_group)

        arm_group = ' '.join(arm_group)
    except:
        arm_group = ''

    try:
        interventions = root.findall('.//intervention')

        # Initialize a list to store the extracted measures
        intervention = []
        # Iterate over each secondary_outcome element
        for i, intervention_entry in enumerate(interventions, start=1):
            i_type_element = intervention_entry.find('intervention_type')
            i_type = f"TYPE: {i_type_element.text}; " if i_type_element is not None else ''

            i_name_element = intervention_entry.find('intervention_name')
            i_name = f"NAME: {i_name_element.text}; " if i_name_element is not None else ''

            i_description_element = intervention_entry.find('description')
            i_description = f"DESCRIPTION: {i_description_element.text}, " if i_description_element is not None else ''
            
            if (i_type != '') | (i_name != '') | (i_description != ''):
                intv = f"INTERVENTION {i}: <{i_type}{i_name}{i_description}>; "
                #intv = f"INTERVENTION {i}: <{i_type}; {i_name}; {i_description}>; "
                intervention.append(intv)

        intervention = ' '.join(intervention)
    except:
        intervention = ''


    if status != 'Completed':
        status = 'Failed'

    data = {'nctid':nctid,
            'phase':phase,
            'status':status,
            'title':title,
            'lead_sponsor':lead_sponsor,
            'brief': brief,
            'description': description,
            'study_design': study_design, # includes intervention information
            'primary_outcome': primary_outcome,
            'secondary_outcome': secondary_outcome,
            'arm_group':arm_group,
            'intervention':intervention,
            'criteria':criteria,
            'enrollment':enrollment}

    return data
    
def GetPhase(phase):
    if phase == 'Phase 1':
        return '1'
    elif phase == 'Phase 2':
        return '2'
    elif phase == 'Phase 3':
        return '3'
    elif phase == 'Phase 1/Phase 2':
        return '2'
    elif phase == 'Phase 2/Phase 3':
        return '3'
    

def PreprocessUnlabledData(nctids_list):
    print('Retrieving unlabled data ...')
    trials_found = 0

    # Add new columns
    column_names = ['NCTID', 'Last_known_Status', 'Last_known_Phase', 'title', 'lead_sponsor', 'brief', 'description', 'study_design', 
                        'outcomes', 'arm_group', 'intervention', 'criteria', 'enrollment'
                    ]
    df = pd.DataFrame(columns=column_names)

    #Parse the XML file for each selected trial and check its completion status
    # If the status isn't 'Completed' we can adjust the phase transition information.
    # If the phase is given, we can directly modify the 'Passed_X' label
        
    # Define the path to the main folder containing subfolders
    main_folder = '../HINT/raw_data/AllPublicXML'

    trials_found = 0
    progress_bar = tqdm(total=TARGET_ENTRIES)

    import random
    selected_files = []

    # Iterate over all subfolders and their files
    for folder_path, _, file_names in tqdm(os.walk(main_folder)):
        # Iterate over all files in the current subfolder
        for file_name in file_names:
            # Check if the file is an XML file
            if file_name.lower().endswith('.xml'):
                filename_without_extension = file_name[:-4]
                #print(filename_without_extension)
                if filename_without_extension in nctids_list:
                    continue
                # Append the file name to the list of selected files
                selected_files.append(os.path.join(folder_path, file_name))

    print(f'Retrived file names: {len(selected_files)}')

    # Randomly select 100,000 files from the list of selected files
    random_selected_files = random.sample(selected_files, TARGET_ENTRIES)

    print(f'Randomly selected files: {len(random_selected_files)}')

    # Iterate over all files in the current subfolder
    for file_name in random_selected_files:
        #print(file_name)
        file_name_only = os.path.basename(file_name)
        nctid = os.path.splitext(file_name_only)[0]
        #print(nctid)

        if nctid in nctids_list:
            print(f"The file {file_name} was already labeled.")
            continue

        data = CheckXML(file_name)

        if data == None:
            continue

        phase = GetPhase(data['phase'])

        #if phase == '':
        #    continue

        # Create a dictionary to hold the new row data
        new_row_data = {
            'NCTID': nctid,
            'Last_known_Status': data['status'],
            'Last_known_Phase': phase,
            'title': data['title'], 
            'lead_sponsor': data['lead_sponsor'], 
            'brief': data['brief'], 
            'description': data['description'],
            'study_design': data['study_design'], 
            #'primary_outcome': data['primary_outcome'],
            #'secondary_outcome': data['secondary_outcome'],
            'outcomes': data['primary_outcome'] + '\n' + data['secondary_outcome'],
            'arm_group': data['arm_group'], 
            'intervention': data['intervention'],  
            'criteria': data['criteria'], 
            'enrollment': data['enrollment'],
        }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            # Append the new row to the DataFrame
            df = df.append(new_row_data, ignore_index=True)
        trials_found += 1
        progress_bar.update(1)

        #if trials_found == TARGET_ENTRIES:
        #    break

    print(f'Entries modified: {trials_found}')

    # Save path file to xml
    #df_unlabled.columns = [col.replace(' ', '_').replace('/', '_') for col in df_unlabled.columns]
    df = df.replace(r'\s+', ' ', regex=True)
    df.to_csv('CThAlpaca/data/raw/unlabled_trial_data.csv', index=False, encoding='utf-8')

def PreprocessLabledData(nctids_list):
    print('Retrieving labled data ...')

    # Add new columns
    column_names = ['NCTID', 'Last_known_Status', 'Last_known_Phase', 'title', 'lead_sponsor', 'brief', 'description', 'study_design', 
                        'outcomes', 'arm_group', 'intervention', 'criteria'
                    ]
    df = pd.DataFrame(columns=column_names)

    # Iterate over all files in the current subfolder
    for nctid in tqdm(nctids_list):
        try:
            current_directory = os.getcwd()
            #print("Current Directory:", current_directory)

            xml_file = '../HINT/raw_data/AllPublicXML/'+nctid[:7]+'xxxx/'+nctid+'.xml'

            data = CheckXML(xml_file)

            if data == None:
                continue

            phase = GetPhase(data['phase'])

            # Create a dictionary to hold the new row data
            new_row_data = {
                'NCTID': nctid,
                'Last_known_Status': data['status'],
                'Last_known_Phase': phase,
                'title': data['title'], 
                'lead_sponsor': data['lead_sponsor'], 
                'brief': data['brief'], 
                'description': data['description'],
                'study_design': data['study_design'], 
                #'primary_outcome': data['primary_outcome'],
                #'secondary_outcome': data['secondary_outcome'],
                'outcomes': data['primary_outcome'] + '\n' + data['secondary_outcome'],
                'arm_group': data['arm_group'], 
                'intervention': data['intervention'],  
                'criteria': data['criteria'], 
                #'enrollment': data['enrollment'],
            }

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                # Append the new row to the DataFrame
                df = df.append(new_row_data, ignore_index=True)

        except FileNotFoundError:
            print(f"The file {xml_file} does not exist.")
            continue

    print(f'Entries modified: {df.shape[0]}')

    # Save path file to xml
    #df_unlabled.columns = [col.replace(' ', '_').replace('/', '_') for col in df_unlabled.columns]
    df = df.replace(r'\s+', ' ', regex=True)
    df.to_csv('CThAlpaca/data/raw/labled_trial_data.csv', index=False, encoding='utf-8')

# Function to transform each entry into the desired format
def create_entry_labled(entry):
    nctid = entry['NCTID']
    sponsor = entry['lead_sponsor']
    description = entry['description']

    # Choose one column name at random
    column_names = ["study_design", "outcomes", "arm_group", "intervention", "criteria"]
    random_column = random.choice(column_names)

    if pd.isna(description):
        #print("Description is empty")
        description = entry['brief']
        random_column = "criteria"

    text = entry[random_column]
    return f"NCTID: {nctid}\nLead Sponsor: {sponsor}\nDescription: {description}\n{random_column}: {text}"

def create_entry_unlabled(entry):
    nctid = entry['NCTID']
    sponsor = entry['lead_sponsor']
    description = entry['description']
    enrollment = entry['enrollment']

    # Choose one column name at random
    column_names = ["study_design", "outcomes", "arm_group", "intervention", "criteria"]
    random_column = random.choice(column_names)

    if pd.isna(description):
        #print("Description is empty")
        description = entry['brief']
        random_column = "criteria"

    text = entry[random_column]
    return f"NCTID: {nctid}\nLead Sponsor: {sponsor}\nenrollment: {enrollment}\nDescription: {description}\n{random_column}: {text}"

def generate_trial_descriptions():

    # Labled data
    df = pd.read_csv('CThAlpaca/data/raw/labled_trial_data.csv')
    print(f'labled_trial_data shape:\n {df.shape}')

    df_labled = pd.DataFrame()

    # Apply the transformation function to each row of the DataFrame
    df_labled['text'] = df.apply(create_entry_labled, axis=1)

    # Save Data
    print(df_labled.head())
    df_labled.to_csv('CThAlpaca/data/processed/labled_trial_text.csv', index=False, encoding='utf-8')  

    # Unlabled data
    df = pd.read_csv('CThAlpaca/data/raw/unlabled_trial_data.csv')
    print(f'unlabled_trial_data shape:\n {df.shape}')

    df_unlabled = pd.DataFrame()

    # Apply the transformation function to each row of the DataFrame
    df_unlabled['text'] = df.apply(create_entry_unlabled, axis=1)

    # Save Data
    print(df_unlabled.head())
    df_unlabled.to_csv('CThAlpaca/data/processed/unlabled_trial_text.csv', index=False, encoding='utf-8')  


def QAPair(df, question_type, instruction, input, response):
    filtered_rows = df.dropna(subset=[input, response])
    df_qa = pd.DataFrame(filtered_rows, columns=[input, response])
    df_qa = df_qa.rename(columns={input: 'input', response: 'response'})
    df_qa['instruction'] = instruction
    df_qa['question_type'] = question_type
    # Reorder columns
    df_qa = df_qa[['question_type', 'instruction', 'input', 'response']]
    return df_qa

'''
Splits all entries into 3 columns: Instruction, Input, Response.
Defines the different tasks 
'''
def CreateQADataset():
    print('Creating QA dataset ...')
    #df = pd.read_csv('data/pretraining/pretraining_dataset.csv')
    df = pd.read_csv('data/processed/CT_phase_transition.csv')
    print(f'pretraining_dataset shape:\n {df.shape}')

    #column_names = ['question_type', 'instruction', 'input', 'response']
    #df_split = pd.DataFrame(columns=column_names)

    # 1. Generation
    # Title to Brief
    question_type = 1
    instruction = 'You are given the title of a clinical trial. Write a brief description that matches the trial title.'
    df_qa_1 = QAPair(df, question_type=question_type, instruction=instruction, input='Trial_Name', response='brief')
    
    df_qa = df_qa_1

    # 2. Summarization
    # Description to Brief
    question_type = 2
    instruction = 'You are given the full description of a clinical trial. Summarize it to short paragraph.'
    df_qa_2 = QAPair(df, question_type=question_type, instruction=instruction, input='description', response='brief')
    df_qa = pd.concat([df_qa, df_qa_2], ignore_index=True)

    # 3. Generation
    # Brief to Intervention
    question_type = 3
    instruction = 'You are given a short description of a clinical trial. Please generate one or multiple interventions that fit the description.'
    #df_qa_3 = QAPair(df, question_type=question_type, instruction=instruction, input='brief', response='intervention')
    #df_qa = pd.concat([df_qa, df_qa_3], ignore_index=True)

    # 4. Generation
    # Description to Inclusion Criteria
    question_type = 4
    instruction = 'You are given the full description of a clinical trial. According to this description, please design inclusion and exclusion criteria for selecting participants.'
    #df_qa_4 = QAPair(df, question_type=question_type, instruction=instruction, input='description', response='criteria')
    #df_qa = pd.concat([df_qa, df_qa_4], ignore_index=True)

    # 5. Numerical Answer
    # Inclusion Criteria to Enrollment
    #df['enrollment'] = df['enrollment'].replace(0.0, np.nan)
    #df['enrollment'] = df['enrollment'].astype('Int64')
    question_type = 5
    instruction = 'You are given the inclusion and exclusion criteria for participant selection in a clinical trial. Please tell me how many participants we will find that fit these criteria.'
    #df_qa_5 = QAPair(df, question_type=question_type, instruction=instruction, input='criteria', response='enrollment')
    #df_qa = pd.concat([df_qa, df_qa_5], ignore_index=True)    

    # 6. Generation
    # Brief to Primary and Secondary Outcome
    #df['outcomes'] = df['primary_outcome'] + '\n' + df['secondary_outcome']
    question_type = 6
    instruction = 'You are given a short description of a clinical trial. Please design one primary outcome for the trial and if needed one or more secondary outcomes.'
    #df_qa_6 = QAPair(df, question_type=question_type, instruction=instruction, input='brief', response='outcomes')
    #df_qa = pd.concat([df_qa, df_qa_6], ignore_index=True)

    # Shuffle
    df_qa = df_qa.sample(frac=1, random_state=42)
    
    # Saving
    df_qa.to_csv('data/pretraining/QA_dataset_labled_data.csv', index=False, encoding='utf-8')  
    print("Q&A dataset succesfully saved to: data/pretraining/QA_dataset.csv ")

# Function to transform each entry into the desired format
def transform_entry_llama(entry):
    instruction = entry['instruction']
    input_text = entry['input']
    response = entry['response']
    return f"<s>[INST]<<SYS>>{instruction}<</SYS>>{input_text}[/INST]{response}</s>"

'''
Creates the prompts in correct textual format from the three columns on which we fine-tune the Llama
'''
def CreateLlamaData():

    df = pd.read_csv('data/pretraining/QA_dataset_labled_data.csv')
    print(f'pretraining_dataset shape:\n {df.shape}')

    df_llama = pd.DataFrame()

    # Apply the transformation function to each row of the DataFrame
    df_llama['text'] = df.apply(transform_entry_llama, axis=1)

    # Save Data
    print(df_llama.head())
    df_llama.to_csv('data/pretraining/ClinicalTrialgov_labled_data_llama.csv', index=False, encoding='utf-8')  
    df_llama.to_parquet('data/pretraining/ClinicalTrialgov_labled_data_llama.parquet', index=False)  


def main():
    # Get NCTIDs
    df_phase_transition = pd.read_csv('data/processed/CT_phase_transition.csv')
    print(f'df_phase_transition shape:\n {df_phase_transition.shape}')
    print(f'df_phase_transition columns:\n {df_phase_transition.columns}')

    nctids_list = df_phase_transition['NCTID'].tolist()

    if PREPROCESS_UNLABLED_DATA:
        PreprocessUnlabledData(nctids_list)

    if PREPROCESS_LABLED_DATA:
        PreprocessLabledData(nctids_list)

    if GENERATE_TRIAL_DESCRIPTIONS:
        generate_trial_descriptions()

    #if CREATE_QA_DATASET:
    #    CreateQADataset()

    #if CREATE_LLAMA_DATA:
    #    CreateLlamaData()

# Check if this script is being run directly
if __name__ == "__main__":
    main()


