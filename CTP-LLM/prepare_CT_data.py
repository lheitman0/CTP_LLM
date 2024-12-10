
import numpy as np
import pandas as pd

PREPROCESS_DATA = True
UPDATE_DATA_BASED_ON_XML_INFO = True
CREATE_DATASET = True

# check xml files for trial outcome
from tqdm import tqdm
from xml.etree import ElementTree as ET
def CheckXML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    nctid = root.find('id_info').find('nct_id').text

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
        primary_outcome = root.find('primary_outcome').find('measure').text
    except:
        primary_outcome = ''

    try:
        secondary_outcome = root.findall('.//secondary_outcome')[0]
        secondary_outcome_0 = secondary_outcome.find('measure').text
    except:
        secondary_outcome_0 = ''

    try:
        secondary_outcome = root.findall('.//secondary_outcome')[1]
        secondary_outcome_1 = secondary_outcome.find('measure').text
    except:
        secondary_outcome_1 = ''

    try:
        healthy_volunteers = root.find('eligibility').find('healthy_volunteers').text
        # print('found criteria')
    except:
        healthy_volunteers = ''

    try:
        gender = root.find('eligibility').find('gender').text
        # print('found criteria')
    except:
        gender = ''

    try:
        # Extract keywords and concatenate into a single string
        keywords = [keyword.text for keyword in root.findall('keyword')]
        keywords_string = ', '.join(keywords)
    except:
        keywords_string = ''

    try:
        allocation = root.find('study_design_info').find('allocation').text
    except:
        allocation = ''

    try:
        intervention_model = root.find('study_design_info').find('intervention_model').text
    except:
        intervention_model = ''

    try:
        primary_purpose = root.find('study_design_info').find('primary_purpose').text
    except:
        primary_purpose = ''

    try:
        masking = root.find('study_design_info').find('masking').text
    except:
        masking = ''


    if status != 'Completed':
        status = 'Failed'

    data = {'nctid':nctid,
           'status':status,
           'phase':phase,
           'criteria':criteria,
           #'enrollment':enrollment,
           'lead_sponsor':lead_sponsor,
           'brief': brief,
           'description': description,
           'primary_outcome': primary_outcome,
           'secondary_outcome_0':secondary_outcome_0,
           'secondary_outcome_1':secondary_outcome_1,
           'healthy_volunteers': healthy_volunteers,
           'gender': gender,
           'keywords': keywords_string,
           'allocation': allocation,
           'intervention_model': intervention_model,
           'primary_purpose': primary_purpose,
           'masking': masking,}

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
    

def PreprocessData():
    # Load drug path
    df_LOA = pd.read_excel('data/raw/Drug_Search_LOA.xls')
    print(f'df_LOA shape:\n {df_LOA.shape}')

    # Drop duplicates from trial information
    df_LOA = df_LOA.drop_duplicates(subset=['DrugIndicationID'])
    print(f'df_LOA shape - no duplicate DrugIndicationIDs:\n {df_LOA.shape}')

    # Only keep drug path information
    df_LOA = df_LOA.drop(columns=['Was in P1','Was in P2','Was in P3','Was in NDA/BLA', 'Current Phase'])
    df_LOA.rename(columns={'Was in P1.1': 'Was in P1',
                            'Was in P2.1': 'Was in P2',
                            'Was in P3.1': 'Was in P3',
                            'Was in NDA/BLA.1': 'Was in NDA/BLA',
                            'Current Phase.1': 'Current Phase'}, inplace=True)
    #print(df_LOA.columns)
    df_path = df_LOA[['DrugIndicationID','Was in P1', 'Was in P2', 'Was in P3', 'Was in NDA/BLA', 'Advance P1',
                        'Advance P2', 'Advance P3', 'Advance NDA/BLA', 'Failed P1', 'Failed P2',
                        'Failed P3', 'Failed NDA/BLA']]
    print(df_path.columns)
    #print(df_path.shape)

    # Kick out all rows that dont hold any information
    columns_to_check = df_path.columns.difference(['DrugIndicationID'])  # Exclude 'DrugIndicationID' from columns to check
    df_path_filtered = df_path.loc[~(df_path[columns_to_check] == 0).all(axis=1)]
    print(f'df_path_filtered shape - only rows with information:\n {df_path_filtered.shape}')

    # Load trial information
    df_TrialSearch = pd.read_csv('data/raw/TrialSearch.csv',encoding='latin-1')
    df_TrialSearch = df_TrialSearch.dropna(subset=['NCTID'])
    print(f'df_Trial_Search shape:\n {df_TrialSearch.shape}')

    # Drop duplicates from trial information
    df_TrialSearch = df_TrialSearch.drop_duplicates(subset=['NCTID'])
    print(f'df_Trial_Search shape - no duplicate NCTIDs:\n {df_TrialSearch.shape}')

    # kick out all rows that don't have a matching DrugIndicationID
    drug_indication_id_list = df_LOA['DrugIndicationID'].tolist()
    df_TrialSearch_filtered = df_TrialSearch[df_TrialSearch['DrugIndicationID'].isin(drug_indication_id_list)]
    print(f'df_Trial_Search shape - only known DrugInIDs:\n {df_TrialSearch_filtered.shape}')

    # only keep Interventional trials
    df_TrialSearch_filtered = df_TrialSearch_filtered[df_TrialSearch_filtered['Trial_Type'].str.contains('intervention', case=False, na=False)]
    print(f'df_Trial_Search shape - only Intervention trials:\n {df_TrialSearch_filtered.shape}') 

    # Add drug path information to the trials
    df_trial_path = pd.merge(df_TrialSearch_filtered, df_path_filtered, on='DrugIndicationID', how='inner')
    print(f'df_trial_path shape:\n {df_trial_path.shape}')

    nctids_list = df_trial_path['NCTID'].tolist()
    entries_modified = 0

    # Add new columns
    new_column_names = ['Last_known_Status', 'Last_known_Phase', 'brief', 'criteria',
                        'description','primary_outcome', 'secondary_outcome_0', 'secondary_outcome_1','Disease_Group','lead_sponsor',
                        'keywords', #'gender', 'healthy_volunteers', 'allocation','intervention_model', 'primary_purpose', 'masking',
                        ]

    # Initialize new columns with None
    for column_name in new_column_names:
        df_trial_path[column_name] = None

    #Parse the XML file for each selected trial and check its completion status
    # If the status isn't 'Completed' we can adjust the phase transition information.
    # If the phase is given, we can directly modify the 'Passed_X' label
    for nctid in tqdm(nctids_list):
        try:
            xml_file = '../../Documents/HINT/raw_data/AllPublicXML/'+nctid[:7]+'xxxx/'+nctid+'.xml'
            data = CheckXML(xml_file)
            phase = GetPhase(data['phase'])
            #print(status)

            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'Last_known_Status'] = data['status']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'Last_known_Phase'] = phase
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'brief'] = data['brief']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'criteria'] = data['criteria']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'description'] = data['description']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'primary_outcome'] = data['primary_outcome']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'secondary_outcome_0'] = data['secondary_outcome_0']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'secondary_outcome_1'] = data['secondary_outcome_1']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'Disease_Group'] = data['Disease_Group']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'lead_sponsor'] = data['lead_sponsor']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'gender'] = data['gender']
            df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'keywords'] = data['keywords']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'healthy_volunteers'] = data['healthy_volunteers']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'allocation'] = data['allocation']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'intervention_model'] = data['intervention_model']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'primary_purpose'] = data['primary_purpose']
            #df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'masking'] = data['masking']

            #val = df_trial_path.loc[df_trial_path['NCTID'] == nctid, 'Last_known_Status'].values[0]
            #print(f'last_known_status: {val}')

        except FileNotFoundError:
            print(f"The file {xml_file} does not exist.")
            continue

    print(f'Entries modified: {entries_modified}')

    # Save path file to xml
    #df_trial_path.columns = [col.replace(' ', '_').replace('/', '_') for col in df_trial_path.columns]
    df_trial_path.to_excel('data/raw/trial_LOA.xls', index=False, encoding='utf-8')

def UpdateData():
    df = pd.read_excel('data/raw/trial_LOA.xls')
    print(f'trial_LOA shape:\n {df.shape}')

    # Fill empty phase information with information from Trial.Search.csv
    print("Number of rows with NaN in 'Last_known_Phase':", df['Last_known_Phase'].isna().sum())

    df['Trial_Phase'] = df['Trial_Phase'].replace({'I': '1', 'II': '2', 'III': '3', 'IIb': '2', 
                                                   'I/II': '2', 'II/III': '3', 'Preclinical': '0',
                                                   'Development': '0', 'IV': '4'})
    df['Last_known_Phase'] = df['Last_known_Phase'].fillna(df['Trial_Phase'])
    print("Number of rows with NaN in 'Last_known_Phase' after augmentation:", df['Last_known_Phase'].isna().sum())

    df.dropna(subset=['Last_known_Phase'], inplace=True)

    failed_condition = (df['Last_known_Status'] == 'Failed') & pd.notna(df['Last_known_Phase'])
    completed_condition = (df['Last_known_Status'] == 'Completed') & pd.notna(df['Last_known_Phase'])
    phase_condition_1 = (df['Was in P1'] == 1) & (df['Last_known_Phase'] == '2')
    phase_condition_2 = (df['Was in P2'] == 1) & (df['Last_known_Phase'] == '3') 
    phase_condition_3 = (df['Was in P1'] == 1) & (df['Was in P2'] == 1) & (df['Last_known_Phase'] == '3')  
    #advance_condition_1 =  (df['Advance P1'] == '1')
    #advance_condition_2 =  (df['Advance P2'] == '1')
    #advance_condition_3 =  (df['Advance P3'] == '1')

    # Update columns based on conditions
    for index, row in tqdm(df.iterrows()):
        if failed_condition[index]:
            phase = int(row['Last_known_Phase'])
            df.at[index, f'Was in P{phase}'] = 1
            df.at[index, f'Advance P{phase}'] = 0
            df.at[index, f'Failed P{phase}'] = 1

        elif completed_condition[index]:
            phase = int(row['Last_known_Phase'])
            df.at[index, f'Was in P{phase}'] = 1

        if phase_condition_1[index]:
            df.at[index, f'Advance P1'] = 1
            df.at[index, f'Failed P1'] = 0

        elif phase_condition_3[index]:
            df.at[index, f'Advance P1'] = 1
            df.at[index, f'Failed P1'] = 0
            df.at[index, f'Advance P2'] = 1
            df.at[index, f'Failed P2'] = 0

        elif phase_condition_2[index]:
            df.at[index, f'Advance P2'] = 1
            df.at[index, f'Failed P2'] = 0

    # Save data
    df.to_excel('data/raw/trial_LOA_updated.xls', index=False, encoding='utf-8')


if PREPROCESS_DATA:
    PreprocessData()

if UPDATE_DATA_BASED_ON_XML_INFO:
    UpdateData()

if CREATE_DATASET:
    df = pd.read_excel('data/raw/trial_LOA_updated.xls')
    print(f'df shape:\n {df.shape}')

    # Drop rows without information
    columns_to_check = ['Advance P1', 'Advance P2', 'Advance P3', 'Advance NDA/BLA', 'Failed P1', 'Failed P2', 'Failed P3', 'Failed NDA/BLA']
    rows_to_drop = df[columns_to_check].eq(0).all(axis=1)

    # Drop the rows where the condition is True
    df = df[~rows_to_drop]
    print(f'df shape after drop:\n {df.shape}')

    # Assuming df is your DataFrame
    count_condition_1 = (df['Was in P1'] == 1) & ((df['Advance P1'] == 1) | (df['Failed P1'] == 1))
    passed_condition_1 = (df['Was in P1'] == 1) & ((df['Advance P1'] == 1))
    failed_condition_1 = (df['Was in P1'] == 1) & ((df['Failed P1'] == 1))

    count_condition_2 = (df['Was in P2'] == 1) & ((df['Advance P2'] == 1) | (df['Failed P2'] == 1))
    passed_condition_2 = (df['Was in P2'] == 1) & ((df['Advance P2'] == 1))
    failed_condition_2 = (df['Was in P2'] == 1) & ((df['Failed P2'] == 1))

    count_condition_3 = (df['Was in P3'] == 1) & ((df['Advance P3'] == 1) | (df['Failed P3'] == 1))
    passed_condition_3 = (df['Was in P3'] == 1) & ((df['Advance P3'] == 1))
    failed_condition_3 = (df['Was in P3'] == 1) & ((df['Failed P3'] == 1))

    print(f"Number of rows P1: {len(df[count_condition_1])}, Passed: {len(df[passed_condition_1])}, Failed: {len(df[failed_condition_1])}")
    print(f"Number of rows P2: {len(df[count_condition_2])}, Passed: {len(df[passed_condition_2])}, Failed: {len(df[failed_condition_2])}")
    print(f"Number of rows P3: {len(df[count_condition_3])}, Passed: {len(df[passed_condition_3])}, Failed: {len(df[failed_condition_3])}")

    # Create the 'Passed_X' columns based on conditions
    df['Passed_I'] = ((df['Advance P1'] == 1) & (df['Failed P1'] == 0)).map({True: 'Yes', False: 'No'})
    df['Passed_II'] = ((df['Advance P2'] == 1) & (df['Failed P2'] == 0)).map({True: 'Yes', False: 'No'})
    df['Passed_III'] = ((df['Advance P3'] == 1) & (df['Failed P3'] == 0)).map({True: 'Yes', False: 'No'})
    df['Approved'] = ((df['Advance NDA/BLA'] == 1) & (df['Failed NDA/BLA'] == 0)).map({True: 'Yes', False: 'No'})

    # Exclude unknown values 
    df.loc[(df['Advance P1'] == 0) & (df['Failed P1'] == 0), 'Passed_I'] = 'Unknown'
    df.loc[(df['Advance P2'] == 0) & (df['Failed P2'] == 0), 'Passed_II'] = 'Unknown'
    df.loc[(df['Advance P3'] == 0) & (df['Failed P3'] == 0), 'Passed_III'] = 'Unknown'
    df.loc[(df['Advance NDA/BLA'] == 0) & (df['Failed NDA/BLA'] == 0), 'Approved'] = 'Unknown'

    pd.set_option('display.max_columns', None)  # Display all columns
    print(df.head())

    file_name = 'data/processed/CT_phase_transition'
    df.to_csv(file_name + '.csv', index=False)
    df.to_excel(file_name + '.xls', index=False, encoding='utf-8')

