import numpy as np
import pandas as pd
import os
import warnings

import hf_olmo

from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")
#message = ["You are given a series of tasks and the description of a clinical trial. Come up with additional tasks that ask questions about the trial, similar to the provided tasks. Description: Introduction: Glass ionomer cements (GICs) are widely used in clinical dentistry due to their advantageous properties. However, they present inferior physical and mechanical properties compared to resin composites. Various techniques have been suggested to improve properties of conventional GICs such as radiant heat transfer by Light Emitting Diode (LED) or lasers, ultrasonic energy transfer also using of (CaCl2) solution. Aim: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. . Methodology: Eighteen healthy patients with 36-second molar teeth will be selected where each patient should have two oclusso- mesial cavities. Standardized oclusso- mesial cavities will be prepared for all the selected teeth, for each patient the first tooth will be restored with chemically cured conventional GICs without any enhancement (M1 group). Meanwhile, the second tooth will be restored by chemically cured conventional GICs that enhanced with radiant heat (LED) (M2 group). functional and biological criteria of each restoration will be clinically evaluated immediately after restoration (T0), six months later (T1), and after 12 months (T2) using Federation Dentaire International (FDI) criteria for assessment of dental restorations. Task 1: Come up with a title for the described clinical trial. Task 2: Is it a randomized study? Task 3: What findings are to be expected? Task 4: Would the trial perform better if only tested on healthy participants?"]
#message = ["You are given the description of a clinical trial and a series of questions realted to the trial. Please answer all questions. Description: Introduction: Glass ionomer cements (GICs) are widely used in clinical dentistry due to their advantageous properties. However, they present inferior physical and mechanical properties compared to resin composites. Various techniques have been suggested to improve properties of conventional GICs such as radiant heat transfer by Light Emitting Diode (LED) or lasers, ultrasonic energy transfer also using of (CaCl2) solution. Aim: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. . Methodology: Eighteen healthy patients with 36-second molar teeth will be selected where each patient should have two oclusso- mesial cavities. Standardized oclusso- mesial cavities will be prepared for all the selected teeth, for each patient the first tooth will be restored with chemically cured conventional GICs without any enhancement (M1 group). Meanwhile, the second tooth will be restored by chemically cured conventional GICs that enhanced with radiant heat (LED) (M2 group). functional and biological criteria of each restoration will be clinically evaluated immediately after restoration (T0), six months later (T1), and after 12 months (T2) using Federation Dentaire International (FDI) criteria for assessment of dental restorations. Task 1: Come up with a title for the described clinical trial.  Answer: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. , Task 2: Is it a randomized study? Answer: , Task 3: What findings are to be expected? Answer: , Task 4: Would the trial perform better if only tested on healthy participants? Answer: ,  Task 5: Would the trial perform better if not randomized? Answer: ,  Task 6: How would you modify the protocol of the trial in terms of treatment arm? Answer: ,  Task 7: What would be a more accurate outcome measure? Answer: ,  Task 8: What questions would you ask if you were conducting the study? Answer: ,  Task 9: What should the primary outcome measure be? Answer: ,  Task 10: What would be a more appropriate primary outcome measure? Answer: ,  Task 11: What would be an appropriate follow-up period? Answer: ,  Task 12: How would you select the sample size? Answer: ,  Task 13: What would be an appropriate power analysis? Answer: ,  Task 14: What would be the null hypothesis? Answer: ,  Task 15: What is the alternative hypothesis? Answer: ,  Task 16: What would be the results of the study? Answer: , "]
message = ["Tell me how common the tuner syndrom is in the US."]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
#inputs = {k: v.to('cuda') for k,v in inputs.items()}
#olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=300, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

'''
You are given a series of tasks and the description of a clinical trial. Come up with additional tasks that 
ask questions about the trial, similar to the provided tasks. 

Description: Introduction: Glass ionomer cements 
(GICs) are widely used in clinical dentistry due to their advantageous properties. However, they present inferior 
physical and mechanical properties compared to resin composites. Various techniques have been suggested to improve 
properties of conventional GICs such as radiant heat transfer by Light Emitting Diode (LED) or lasers, ultrasonic 
energy transfer also using of (CaCl2) solution. Aim: Clinical evaluation of chemically cured conventional glass 
ionomer after light-emitting diode radiant heat enhancement.. Methodology: Eighteen healthy patients with 36-second 
molar teeth will be selected where each patient should have two oclusso- mesial cavities. Standardized oclusso-mesial 
cavities will be prepared for all the selected teeth, for each patient the first tooth will be restored with chemically 
cured conventional GICs without any enhancement (M1 group). Meanwhile, the second tooth will be restored by chemically 
cured conventional GICs that enhanced with radiant heat (LED) (M2 group). functional and biological criteria of each restoration 
will be clinically evaluated immediately after restoration (T0), six months later (T1), and after 12 months (T2) 
using Federation Dentaire International (FDI) criteria for assessment of dental restorations. 

Task 1: Come up with a title for the described clinical trial. 
Task 2: Is it a randomized study? 
Task 3: What findings are to be expected? 
Task 4: Would the trial perform better if only tested on healthy participants? 
Task 5: Would the trial perform better if not randomized? 
Task 6: How would you modify the protocol of the trial in terms of treatment arm? 
Task 7: What would be a more accurate outcome measure? 
Task 8: What questions would you ask if you were conducting the study? 
Task 9: What should the primary outcome measure be? 
Task 10: What would be a more appropriate primary outcome measure? 
Task 11: What would be an appropriate follow-up period? 
Task 12: How would you select the sample size? 
Task 13: What would be an appropriate power analysis? 
Task 14: What would be the null hypothesis? 
Task 15: What is the alternative hypothesis? 
Task 16: What would be the results of the study? 
Task 17: Are there possible reasons to reject the null hypothesis? 
Task 18: Would these results be of clinical significance? 
Task 19: What is the significance level of the statistical analysis? 
Task 20: What statistical method would you use? 
Task 21: What statistical tests should be run? 
Task 22: What would be an appropriate sample size? 
Task 23: What type of sampling method would you use? 
Task 24: What would be the appropriate statistical tests to use in a randomized clinical trial? 
Task 25: What would be the appropriate statistical test in this case? 
Task 26: What would be the appropriate measure of effect? 
Task 27: What would be the appropriate effect size? 
Task 28: What is the level of significance in a statistical test? 
Task 29: What would be the appropriate power in this study? 
Task 30: What would be the appropriate mean in this study? 
Task 31: What is the standard deviation in this study? 
Task 32: What would be an appropriate confidence interval for the mean of this study? 
Task 33: What would be an appropriate mean for this study?
'''