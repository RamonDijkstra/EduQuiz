This repository contains the data and code for the experiments in the paper: Reading Comprehension Quiz Generation using Generative Pre-trained Transformers.

# Experiments

Experiment 1: Step-Wise Quiz Generation (SWQG)
1) prompt: context ------------------------------------------> completion: question    
2) prompt: context, generated question ---------------------> completion: answer      
3) prompt: context, generated question, generated answer --> completion: distractors 

note: experiment 1 concatenates completion 1, 2, and 3

Experiment 2: End-to-End Quiz Generation (EEQG)
- prompt: context
- completion: question, answer, and distractors

# What is in the folders?

    - RACE: Original data from RACE (https://www.cs.cmu.edu/~glai1/data/race/)
    - generated_data_gpt3: Contains the ground-truth data, generations, and automatic evaluation scores from GPT-3
    - generated_data_macaw: Contains the ground-truth data, generations, and automatic evaluation scores from Macaw-11b
    - gpt3_completion_scripts: Scripts to perform completion on the test data with GPT-3
    - gpt3_costs: Costs to run the experiments with GPT-3
    - gpt3_evaluation_scripts: Scripts to evaluate the experiments on the test data with GPT-3
    - human_evaluation: All test instances where we performed human evaluation
    - key-race: Original data from EQG-RACE (https://github.com/jemmryx/EQG-RACE)
    - macaw_evaluation_scripts: Scripts to evaluate the experiments on the test data with Macaw-11b
    - processed_data: Combined EQG-RACE with original RACE data
    - gpt3_finetune_instructions.txt: Instructions to fine-tune GPT-3
    - macaw_completion.py: Script to perform completion on the test data with Macaw-11b
    - preprocess_data.py: Script to combine EQG-RACE with original RACE data


# How to reproduce the results?

    - Fine-tune the GPT-3 models with the training/validation files from the processed_data directory using the instructions in gpt3_finetune_instructions.txt
    - Run all gpt3_completion_scripts with the created fine-tuned GPT-3 models
    - Run the macaw_completion.py script
    - Run all gpt3_evaluation_scripts
    - Run all macaw_evaluation_scripts
    
note: look carefully which files are required for which completion/evaluation.

# Disclaimer
The use of transformer models, especially large pre-trained language models comes with the risk of using inappropriate language. As pre-trained language models are trained on the humanly created text which could contain inappropriate language, these features also come back during inference time. For every application with the proposed methods, we recommend using the models carefully. In this way we can make use of the advantages that come with these models, thereby staying critical to the outcome.
