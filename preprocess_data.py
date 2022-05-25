import os
import json
import random

def get_data_instances(data_path, eqg_processed, completion_option):
    '''
    Input: 
    - data_path (can be train_data, validation_data, test_data)
        
    - completion_option = 1 -> prompt: context                                   completion: question
    - completion_option = 2 -> prompt: context, question                         completion: answer
    - completion_option = 3 -> prompt: context, question, answer                 completion: distractors
    - completion_option = 4 -> prompt: context                                   completion: question, answer, and distractors
    
    
    Output:
    Two lists with the following dictionaries
    
    1 {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    2 {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    3 {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    4 ...
    
    Count of all data instances
    Count of relevant data instances
    
    '''
    
    gpt_data_list = []
    macaw_data_list = []
    total_count = 0
    relevant_count = 0
    
    for filename in os.listdir(data_path):
        with open(data_path + filename) as f:
            data_instance = json.load(f)
            
            # Extract the dictionary
            article = data_instance['article']
            questions = data_instance['questions']
            options = data_instance['options']
            answers = data_instance['answers']
            id_ = data_instance['id']

            # As there are multipe questions per article
            for i in range(len(questions)):
                # Count total instances
                total_count += 1
                question = questions[i]
                compare_question = question.lower().replace(" ","")
                
                # Only process the questions that are in the EQG-RACE dataset
                if len(eqg_processed) != 0 and compare_question == eqg_processed[0]:
                    eqg_processed.pop(0)
                    
                    relevant_count += 1
                    
                    # Select the true answer and distractors
                    if answers[i] == 'A':
                        true_answer = options[i][0]
                        distractors = [distractor for distractor in options[i] if distractor != true_answer]
                        
                    elif answers[i] == 'B':
                        true_answer = options[i][1]
                        distractors = [distractor for distractor in options[i] if distractor != true_answer]
                        
                    elif answers[i] == 'C':
                        true_answer = options[i][2]
                        distractors = [distractor for distractor in options[i] if distractor != true_answer]
                        
                    else:
                        true_answer = options[i][3]
                        distractors = [distractor for distractor in options[i] if distractor != true_answer]
                 
                    # Different completion options       
                    if completion_option == 1:
                        gpt_prompt_text = article + '\n\n###\n\n'
                        gpt_completion = ' ' + 'Question: ' + question + '\n###'
                        
                        macaw_prompt_text = "$question$" + " ; " + "$context$ = " + article
                        macaw_completion = question

                    elif completion_option == 2:
                        gpt_prompt_text = article + '\n\n' + 'Question: ' + question + '\n\n###\n\n'
                        gpt_completion = ' ' + 'Answer: ' + true_answer + '\n###'
                        
                        macaw_prompt_text = "$answer$" + " ; " + "$question$ = " + question + " $context$ = " + article
                        macaw_completion = true_answer

                    elif completion_option == 3:
                        gpt_prompt_text = article + '\n\n' + 'Question: ' + question + '\n\n' + 'Answer: ' + true_answer + '\n\n###\n\n'
                        gpt_completion = ' '
                        for distractor in distractors:
                            gpt_completion += 'False answer: ' + distractor + '\n'
                        gpt_completion += '###'
                            
                        macaw_prompt_text = "$mcoptions$" + " ; " + "$question$ = " + question +  " $answer$ = " + true_answer + " $context$ = " + article
                        macaw_completion = ''
                        for distractor in distractors:
                            macaw_completion += distractor + '\n'

                    elif completion_option == 4:
                        gpt_prompt_text = article + '\n\n###\n\n'
                        gpt_completion = ' ' + 'Question: ' + question + '\n' + 'True answer: ' + true_answer + '\n'
                        for distractor in distractors:
                            gpt_completion += 'False answer: ' + distractor + '\n'
                        gpt_completion += '###'
                        
                        macaw_prompt_text = "$question$ ; $answer$ ; $mcoptions$ ;" + " $context$ = " + article
                        macaw_completion = 'Question: ' + question + '\n' + 'True answer: ' + true_answer + '\n'
                        for distractor in distractors:
                            macaw_completion += 'False answer: ' + distractor + '\n'

                    else:
                        assert False, 'Quiz completion options are: 1,2,3,4'
                    
                    # Put the data instances in a dict
                    gpt_dict = {"prompt": gpt_prompt_text, "completion": gpt_completion}
                    gpt_data_list.append(gpt_dict)
                        
                    macaw_dict = {"prompt": macaw_prompt_text, "completion": macaw_completion}
                    macaw_data_list.append(macaw_dict)                 

    return gpt_data_list, macaw_data_list, total_count, relevant_count

def data_preprocess(completion_option):
    # Get relevant questions from EQG-RACE
    eqg_train_questions_processed, eqg_val_questions_processed, eqg_test_questions_processed = process_eqg()
    
    # Get data instances for all data paths
    gpt_processed_train_data_high, _, train_total_count_h, train_relevant_count_h = get_data_instances(train_high, eqg_train_questions_processed, completion_option=completion_option)
    gpt_processed_train_data_middle, _, train_total_count_m, train_relevant_count_m = get_data_instances(train_middle, eqg_train_questions_processed, completion_option=completion_option)

    gpt_processed_val_data_high, _, val_total_count_h, val_relevant_count_h = get_data_instances(val_high, eqg_val_questions_processed, completion_option=completion_option)
    gpt_processed_val_data_middle, _, val_total_count_m, val_relevant_count_m = get_data_instances(val_middle, eqg_val_questions_processed, completion_option=completion_option)

    gpt_processed_test_data_high, macaw_processed_test_data_high, test_total_count_h, test_relevant_count_h = get_data_instances(test_high, eqg_test_questions_processed, completion_option=completion_option)
    gpt_processed_test_data_middle, macaw_processed_test_data_middle, test_total_count_m, test_relevant_count_m = get_data_instances(test_middle, eqg_test_questions_processed, completion_option=completion_option)
    
    # Concatentate high and middle school data
    gpt_train_data = gpt_processed_train_data_high + gpt_processed_train_data_middle
    gpt_val_data = gpt_processed_val_data_high + gpt_processed_val_data_middle
    gpt_test_data = gpt_processed_test_data_high + gpt_processed_test_data_middle

    macaw_test_data = macaw_processed_test_data_high + macaw_processed_test_data_middle
    
    # Get counts
    train_total_count = train_total_count_h + train_total_count_m
    val_total_count = val_total_count_h + val_total_count_m
    test_total_count = test_total_count_h + test_total_count_m
    
    train_relevant_count = train_relevant_count_h + train_relevant_count_m
    val_relevant_count = val_relevant_count_h + val_relevant_count_m
    test_relevant_count = test_relevant_count_h + test_relevant_count_m
    
    # Only print this one time.
    if completion_option == 1: 
        print('Total training instances, relevant instances and percentage', train_total_count, train_relevant_count, round(train_relevant_count/train_total_count * 100, 2), '%')
        print('Total validation instances, relevant instances and percentage', val_total_count, val_relevant_count, round(val_relevant_count/val_total_count * 100, 2), '%')
        print('Total test instances, relevant instances and percentage', test_total_count, test_relevant_count, round(test_relevant_count/test_total_count * 100, 2), '%')
        
    # Shuffle the train data
    random.shuffle(gpt_train_data)
      
    return gpt_train_data, gpt_val_data, gpt_test_data, macaw_test_data

def process_eqg():
    # Open the EQG files
    with open('./key-race/train.json') as f:
        train_eqg = json.load(f)

    with open('./key-race/dev.json') as f:
        val_eqg = json.load(f)

    with open('./key-race/test.json') as f:
        test_eqg = json.load(f)
        
    # Extract the questions from the json files
    train_questions = []
    val_questions = []
    test_questions = []

    for train_instance in train_eqg:
        train_questions.append(train_instance['question'])

    for val_instance in val_eqg:
        val_questions.append(val_instance['question'])

    for test_instance in test_eqg:
        test_questions.append(test_instance['question'])
        
    # Remove the additional spaces so we can perform a string comparison with RACE data
    eqg_train_questions_processed = []
    eqg_val_questions_processed = []
    eqg_test_questions_processed = []
    
    for question in train_questions:
        question = question.replace(" ", "")
        eqg_train_questions_processed.append(question)
   
    for question in val_questions:
        question = question.replace(" ", "")
        eqg_val_questions_processed.append(question)

    for question in test_questions:
        question = question.replace(" ", "")
        eqg_test_questions_processed.append(question)
        
    return eqg_train_questions_processed, eqg_val_questions_processed, eqg_test_questions_processed
    
def get_filenames_race():
    train_high = './RACE/train/high/'
    train_middle = './RACE/train/middle/'

    val_high = './RACE/dev/high/'
    val_middle = './RACE/dev/middle/'

    test_high = './RACE/test/high/'
    test_middle = './RACE/test/middle/'
    
    return train_high, train_middle, val_high, val_middle, test_high, test_middle

def write_files():
    # Have all the 4 different completion options
    completion_options = [1,2,3,4]

    for completion_option in completion_options:
        # Get the data
        gpt_train_data, gpt_val_data, gpt_test_data, macaw_test_data = data_preprocess(completion_option)
        
        # Create the directories
        path = os.path.abspath(os.getcwd())
    
        os.makedirs(path + '/processed_data/' + 'gpt3/completion_' + str(completion_option))
        os.makedirs(path + '/processed_data/' + 'macaw/completion_' + str(completion_option))

        # Write the data
        # GPT-3
        with open('./processed_data/' + 'gpt3/completion_' + str(completion_option) + "/processed_train.json", 'w') as f:
            for item in gpt_train_data:
                f.write(json.dumps(item) + "\n")

        with open('./processed_data/' + 'gpt3/completion_' + str(completion_option) + "/processed_val.json", 'w') as f:
            for item in gpt_val_data:
                f.write(json.dumps(item) + "\n")
                        
        with open('./processed_data/' + 'gpt3/completion_' + str(completion_option) + "/processed_test.json", 'w') as f:
            for item in gpt_test_data:
                f.write(json.dumps(item) + "\n")   

        # Macaw
        with open('./processed_data/' + 'macaw/completion_' + str(completion_option) + "/processed_test.json", 'w') as f:
            for item in macaw_test_data:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    # Get filenames of RACE
    train_high, train_middle, val_high, val_middle, test_high, test_middle = get_filenames_race()
    
    # Set random seed
    random.seed(7)

    # Write files
    write_files()