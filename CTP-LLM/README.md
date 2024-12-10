# Download ClinialTrials.gov data

##### * Download data

mkdir -p raw_data
cd raw_data
wget https://clinicaltrials.gov/AllPublicXML.zip # This will take 10-20 minutes to download

##### * Unzip ZIP file

unzip AllPublicXML.zip # This might take over an hour to run, depending on your system
cd ../

##### * Collect and sort all the XML files and put output in all_xml

find raw_data/ -name NCT*.xml | sort > data/all_xml
head -3 data/all_xml

##### * Remove ZIP file to recover some disk space

rm raw_data/AllPublicXML.zip

# Creating the Dataset

1. Download [data](https://www.dropbox.com/scl/fo/beg0bd1tvhs3vr59yivnz/AAVx8wuxdIWLiHuf6NgfdQ0?rlkey=oa02zadqk1jh5y16csj70u7w4&st=8warw3ji&dl=0) and put files into 'data/raw'.
2. Open 'prepare_CT_data.py'.
3. Specify path to ClinicalTrial XML files in line 223.
4. Before run, set options (if first run, set all three to 'True').

##### Options:

###### Line 5 - PRECPROCESS_DATA

Extracts raw data from ClinicalTrials XML files. It only uses entries that are also present in 'Drug_Search_LOA.xls'. Creates the file 'data/raw/trial_LOA.xls' in which it saves the found entries.

###### Line 6 - UPDATE_DATA_BASED_ON_XML_INFO

Opens file 'data/raw/trial_LOA.xls' and cleans data. Takes first steps to create accurate label. Saves file to 'data/raw/trial_LOA_updated.xls'.

##### Line 7 - CREATE_DATASET

Opens file 'data/raw/trial_LOA_updated.xls'. Applies our labelling system to data and saves it as 'data/processed/Phase_Transition_Dataset.csv' - the final dataset.

# Split Dataset by Date to prevent Bias

1. Open 'split_on_date.py'.
2. Split index (date) specified in line 22
3. Run

Use resulting files respectively when creating train or test sets.

# Create Training/Testing Files

1. Open 'data_preprocessor.py'.
2. To create train and validation sets use 'data/train_PT.csv' in line 37, else 'data/test_PT.csv'.
3. Specify options for the specific dataset you want to create by changing the Parameters (lines 28-32).
4. Run file.
5. Creates .csv, .json, and .parquet (needed for LLaMA) files.
6. Open 'GPT_fine_tuning/CT_prediction_fine_tuning.py'.
7. Specify the previously created .csv file in line 7.
8. Run.
9. Creates .csv for BERT-RF and Longformer training and .jsonl files needed for CTP-LLM (GPT 3.5) fine-tuning.

# BERT-RF

1. Open 'BERT_RF/decode_file.py'.
2. Execute the code for train, val, and test .csv files to prepare them for BERT processing. Creates decoded files.
3. Open 'BERT_RF/bert_encoder.py'.
4. Execute the code on decoded train, val, and test files to encode them with BERT. Creates encoded files.
5. Open 'BERT_RF/random_forest.py'
6. Specify path to encoded train, val, and test files in lines 22-24.
7. Run to train and evaluate BERT-RF.

# CTP-LLM (GTP 3.5)

1. Go to [OpenAI Fine Tuning](https://platform.openai.com/finetune) and log in to your account.
2. Create a new fine tuning job.
3. Select gpt-3.5-turbo-0125 as model.
4. Upload training and validation data (both .jsonl files created previously).
5. Specify parameters and start job (be aware that this can cost a significant amount of money).
6. For evaluation of the fine-tuned model open 'GPT_fine_tuning/evaluate_model.py'.
7. Provide the model key ('ft:gpt-3.5-turbo-0125:personal: ...') to the MODEL parameter (line 9).
8. Enter the API key you received when creating your account or project in API_KEY (line 14).
9. Specify the path to the .jsonl test file in FILE_PATH (line 11).
10. Run

# Train Longformer

1. Open 'baseline/longformer/longformer.py'.
2. Specify train and val files in lines 19 and 25 respectively (use .csv).
3. When running provide wandb credentials (create an account if you don't have any).

# LLaMA 2 7B

1. Open 'baseline/LLaMA/Fine_tune_Llama_2.ipynb'.
2. Execute steps 1 & 2 (if torch cannot be imported reconnect notebook).
3. Step 3: Specify dataset name in line 6 (dataset has to come from huggingface).
4. Depending on the selected GPU on Colab change 'fp16' and 'bf16' settings in lines 51 and 52.
5. Step 4: Execute first cell to import datasets.
6. Execute following cell to sart model training (can take several hours). Don't forget to save the new model with the following cell.
7. Step 5: Execute to visualize model training.
8. Step 6: Execute first cell to initilaize functions for following steps.
9. If you just fine-tuned a new model ignore the "Load pretrained model" cell. If you want to work with an older model import it here before evaluation.
10. Execute the following cells to import the true labels and create the predicted labels. The 'evaluate()' function will compare y_true and y_pred to compute F1 and Accuracy.
11. Steps 7 and 8 store the model and push it to your huggingface account.
