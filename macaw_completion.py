from utils import load_model, run_macaw
from tqdm import tqdm
import json
import os

model_dict = load_model("allenai/macaw-11b")

# SWQG

path = "processed_data/macaw/completion_1/processed_test.json"

test_data = []

for line in open(path):
    test_data.append((json.loads(line)))

generated = {}

for i in tqdm(range(len(test_data))):
    res = run_macaw(test_data[i]['prompt'], model_dict)
    output_raw_list = res['output_raw_list']    
    generated[i+1] = output_raw_list

with open("generated_data_macaw/SWQG/generated_questions.json", 'w') as f:
    json.dump(generated, f)

print("Experiment 1.1 done")

questions = []

for key in generated:
    q = generated[key][0].split("= ")[1].split(". ")[-1]
    questions.append(q)

generated = {}

for i in tqdm(range(len(test_data))):
    prompt = test_data[i]['prompt']
    context = prompt.split("$question$ ;")[1]
    new_prompt = "$answer$ ; " + "$question$ = " + questions[i] + context
    res = run_macaw(new_prompt, model_dict)
    output_raw_list = res['output_raw_list']    
    generated[i+1] = output_raw_list

with open("generated_data_macaw/SWQG/generated_answers.json", 'w') as f:
    json.dump(generated, f)

print("Experiment 1.2 done")

answers = []
for key in generated:
    a = generated[key][0].split("= ")[1].split(". ")[-1]
    answers.append(a)

generated = {}

for i in tqdm(range(len(test_data))):
    prompt = test_data[i]['prompt']
    context = prompt.split("$question$ ;")[1]
    new_prompt = "$mcoptions$ ; " + "$question$ = " + questions[i] + " $answer$= " + answers[i] + context
    res = run_macaw(new_prompt, model_dict)
    output_raw_list = res['output_raw_list']    
    generated[i+1] = output_raw_list

with open("generated_data_macaw/SWQG/generated_distractors.json", 'w') as f:
    json.dump(generated, f)

print("Experiment 1.3 done")

# EEQG

path = "processed_data/macaw/completion_4/processed_test.json"

test_data = []

for line in open(path):
    test_data.append((json.loads(line)))

generated = {}

for i in tqdm(range(len(test_data))):
    res = run_macaw(test_data[i]['prompt'], model_dict)
    output_raw_list = res['output_raw_list']    
    generated[i+1] = output_raw_list

with open("generated_data_macaw/EEQG/generated_quiz.json", 'w') as f:
    json.dump(generated, f)

print("Experiment 2 done")