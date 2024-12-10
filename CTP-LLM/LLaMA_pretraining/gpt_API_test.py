from openai import OpenAI
import os

#client.api_key = ['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']

client = OpenAI(
   api_key='api_key',
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {"role": "system", "content": "You are an expert in the medical domain, especially clinical trial design and analysis."},
    #{"role": "user", "content": "You are given the description of a clinical trial and a series of questions realted to the trial. Please answer all questions. Description: Introduction: Glass ionomer cements (GICs) are widely used in clinical dentistry due to their advantageous properties. However, they present inferior physical and mechanical properties compared to resin composites. Various techniques have been suggested to improve properties of conventional GICs such as radiant heat transfer by Light Emitting Diode (LED) or lasers, ultrasonic energy transfer also using of (CaCl2) solution. Aim: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. . Methodology: Eighteen healthy patients with 36-second molar teeth will be selected where each patient should have two oclusso- mesial cavities. Standardized oclusso- mesial cavities will be prepared for all the selected teeth, for each patient the first tooth will be restored with chemically cured conventional GICs without any enhancement (M1 group). Meanwhile, the second tooth will be restored by chemically cured conventional GICs that enhanced with radiant heat (LED) (M2 group). functional and biological criteria of each restoration will be clinically evaluated immediately after restoration (T0), six months later (T1), and after 12 months (T2) using Federation Dentaire International (FDI) criteria for assessment of dental restorations. Q 1: Come up with a title for the described clinical trial.  Answer: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. , Q 2: Is it a randomized study? Answer: , Q 3: What findings are to be expected?, Q 4: Would the trial perform better if only tested on healthy participants? Answer: ,  Q 5: Would the trial perform better if not randomized? Answer: ,  Q 6: How would you modify the protocol of the trial in terms of treatment arm? Answer: ,  Q 7: What would be a more accurate outcome measure? Answer: ,  Q 8: What questions would you ask if you were conducting the study? Answer: ,  Q 9: What should the primary outcome measure be? Answer: ,  Q 10: What would be a more appropriate primary outcome measure? Answer: ,  Q 11: What would be an appropriate follow-up period? Answer: ,  Q 12: How would you select the sample size? Answer: ,  Q 13: What would be an appropriate power analysis? Answer: ,  Q 14: What would be the null hypothesis? Answer: ,  Q 15: What is the alternative hypothesis? Answer: ,  Q 16: What would be the results of the study? Answer: , "}
    {"role": "user", "content": "You are given the description of a clinical trial and a series of questions realted to the trial. Please answer all questions. Description: Introduction: Glass ionomer cements (GICs) are widely used in clinical dentistry due to their advantageous properties. However, they present inferior physical and mechanical properties compared to resin composites. Various techniques have been suggested to improve properties of conventional GICs such as radiant heat transfer by Light Emitting Diode (LED) or lasers, ultrasonic energy transfer also using of (CaCl2) solution. Aim: Clinical evaluation of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement. . Methodology: Eighteen healthy patients with 36-second molar teeth will be selected where each patient should have two oclusso- mesial cavities. Standardized oclusso- mesial cavities will be prepared for all the selected teeth, for each patient the first tooth will be restored with chemically cured conventional GICs without any enhancement (M1 group). Meanwhile, the second tooth will be restored by chemically cured conventional GICs that enhanced with radiant heat (LED) (M2 group). functional and biological criteria of each restoration will be clinically evaluated immediately after restoration (T0), six months later (T1), and after 12 months (T2) using Federation Dentaire International (FDI) criteria for assessment of dental restorations.; Task 5: How will the patients be monitored for any potential adverse effects or complications related to the use of chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement?\n\nTask 6: What specific criteria will be used to assess the physical and mechanical properties of the restorations at each evaluation time point (T0, T1, and T2)?\n\nTask 7: How will the clinicians ensure standardization in the preparation of oclusso-mesial cavities for the study participants?\n\nTask 8: How will the patients be instructed to care for their restorations between evaluation time points to ensure accurate clinical assessment?\n\nTask 9: How will the researchers account for factors such as patient age and oral hygiene habits in the evaluation of the restorations over the 12-month period?\n\nTask 10: Will the researchers consider conducting additional analyses, such as patient satisfaction surveys or cost-effectiveness assessments, to provide a more comprehensive evaluation of the chemically cured conventional glass ionomer after light-emitting diode radiant heat enhancement compared to conventional GICs without enhancement? "}
  ]
)

print(completion.choices[0].message)

'''
ChatCompletionMessage(
    content=
        'Q 1: Title for the trial: "Evaluation of Light-Emitting Diode Radiant Heat Enhancement on Chemically Cured 
            Conventional Glass Ionomer Restorations in Molar Teeth"\n\n
        Q 2: Yes, this study should ideally be a randomized controlled trial to ensure unbiased allocation of treatment.\n\n
        Q 3: The expected findings include improved physical and mechanical properties of glass ionomer restorations with 
            LED radiant heat enhancement compared to conventional GIC restorations.\n\n
        Q 4: Testing on healthy participants is preferred in order to eliminate confounding factors that could affect the trial results.\n\n
        Q 5: Randomization is crucial in order to minimize selection bias and ensure comparability between treatment groups. 
            Not randomizing the trial could result in skewed results.\n\n
        Q 6: If modifying the treatment arm, you could consider adding a third group that receives a different enhancement 
            technique, or comparing the LED enhancement to other enhancement methods.\n\n
        Q 7: A more accurate outcome measure could involve quantitative assessments of physical and mechanical properties of the 
            restorations along with the clinical evaluation.\n\n
        Q 8: Questions to ask during the study could include assessing patient comfort during and after the restoration, 
            evaluating any adverse events related to the treatment, and getting feedback on the aesthetics of the restoration.\n\n
        Q 9: The primary outcome measure should be the clinical evaluation of the restorations based on the FDI criteria at T0, T1, and T2.\n\n
        Q 10: An appropriate primary outcome measure could be the retention rate of the restorations over the follow-up period.\n\n
        Q 11: An appropriate follow-up period would be at least 12 months to evaluate the long-term performance and durability of the restorations.\n\n
        Q 12: Sample size should be determined based on power analysis to ensure the study has enough statistical power to 
            detect a clinically significant difference between the treatment groups.\n\nQ 13: An appropriate power analysis would calculate 
            the minimum sample size required to detect a specified effect size with a certain level of confidence.\n\n
        Q 14: Null hypothesis: There is no significant difference in the clinical outcomes of chemically cured conventional 
            GIC restorations with and without LED radiant heat enhancement.\n\nQ 15: Alternative hypothesis: Chemically cured conventional 
            GIC restorations with LED radiant heat enhancement demonstrate superior clinical outcomes compared to 
            conventional GIC restorations without enhancement.\n\n
        Q 16: The results are expected to show that the LED radiant heat enhancement improves the clinical performance 
            and longevity of glass ionomer restorations in molar teeth over the follow-up period.', 
    role='assistant', function_call=None, tool_calls=None)
'''