# This script retrieves trial information from ClinicalTrial.gov
# Its purpose is to fine-tune the LLM on CT related geration tasks,
# before training it on our phase transition prediction task.

import numpy as np
import pandas as pd
import os
import warnings
import random

# check xml files for trial outcome
from tqdm import tqdm
from xml.etree import ElementTree as ET
def CheckXML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    nctid = root.find('id_info').find('nct_id').text

    try:
        why_stopped = root.find('why_stopped').text 
    except:
        return None

    

    data = {'nctid':nctid,
            'why_stopped':why_stopped}

    return data

    

def get_why_stopped_information(nctids_list):
    print('Retrieving why_stopped information ...')

    # Add new columns
    column_names = ['NCTID', 'why_stopped']
    df = pd.DataFrame(columns=column_names)

    # Iterate over all files in the current subfolder
    for nctid in tqdm(nctids_list):
        try:
            #current_directory = os.getcwd()
            #print("Current Directory:", current_directory)

            xml_file = '../../Documents/HINT/raw_data/AllPublicXML/'+nctid[:7]+'xxxx/'+nctid+'.xml'

            data = CheckXML(xml_file)

            if data == None:
                continue

            # Create a dictionary to hold the new row data
            new_row_data = {
                'NCTID': nctid,
                'why_stopped':data['why_stopped']
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
    print(df)

    return df
    #df.to_csv('GPT_fine_tuning/data/processed/why_stopped.csv', index=False, encoding='utf-8')


def main():
    # Get NCTIDs
    df_phase_transition = pd.read_csv('data/raw/CT_phase_transition.csv')
    print(f'df_phase_transition shape:\n {df_phase_transition.shape}')
    print(f'df_phase_transition columns:\n {df_phase_transition.columns}')

    nctids_list = df_phase_transition['NCTID'].tolist()

    df_why_stopped = get_why_stopped_information(nctids_list)

    merged_df = pd.merge(df_phase_transition, df_why_stopped, on='NCTID', how='outer')

    merged_df.to_csv('GPT_fine_tuning/data/processed/CT_reasoning.csv', index=False, encoding='utf-8')

# Check if this script is being run directly
if __name__ == "__main__":
    main()


