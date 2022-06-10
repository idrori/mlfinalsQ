import openai 
import pandas as pd
import time
import argparse
import os
from embedding import get_embeddings, get_most_similar

openai.api_key = os.getenv('OpenAI_API_Key')
all_finals = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']

parser = argparse.ArgumentParser()
# if an argument is passed in as True, we do it
parser.add_argument("--Codex_Few_Shot")
parser.add_argument("--GPT3_CoT_Few_Shot")
parser.add_argument("--GPT3_Few_Shot")
args = parser.parse_args()

#Will use this many few-shot examples if possible: (if fewer are solved, use as many as possible)
few_shot_examples_desired = 1
codex_engine = "code-davinci-002"
gpt3_engine = "text-davinci-002"
engine_temperature = 0
engine_topP = 0
few_shot_max_tokens = 1024
gpt3_max_tokens = 500
gpt3_CoT_max_tokens = 1000
codex_time_delay = 3
gpt3_time_delay = 1
CoT = "Let's think step by step."

def execute_few_shot(finals):
    """
    Runs Codex few-shot on questions_per questions for each course in courses.
    """
    fewshot_embeddings = [] # not sure about this yet
    for final in finals:
        final_location = 'results/' + final + ' results.csv'
        #initializing new columns in csv
        results = pd.read_csv(final_location)
        results['Few-Shot Input'] = ''
        results['Few-Shot Output'] = ''
        results['Few-Shot Evaluation'] = ''
        results.to_csv(final_location, index=False)

        #getting all qs and their embeddings
        final_qs = []
        final_embeddings = get_embeddings('data/embeddings/' + final + '_embeddings.json')
        for i in range(len(results['Question'])):
            question = results.loc[i, 'Question']
            final_qs.append(question)     
        if  final != all_finals[0]:
            #doing few shot for every question
            for i in range(len(final_qs)):
                k = few_shot_examples_desired

                #correct via zero-shot:
                if results.iloc[i]['Zero-Shot Evaluation'] == 1:
                    print('no few shot needed for ' + final + ' question ' + str(i+1))
                    few_shot_input = 'n/a'
                    few_shot_output = 'n/a'

                #incorrect via zero-shot:
                elif 0 <= results.iloc[i]['Zero-Shot Evaluation'] < 1:
                    few_shot_input = ''
                    print('doing Codex few-shot for ' + final + ' question ' + str(i+1) + '...')

                    #to find the candidate questions information and use it if it works
                    for closest in get_most_similar(fewshot_embeddings, final_embeddings[i]):
                        for dif_final in finals:
                            embeddings = get_embeddings('data/embeddings/' + dif_final + '_embeddings.json')
                            if closest > len(embeddings):
                                closest -= len(embeddings)
                            else:
                                index = closest - 1
                                print(f'found closest question in {dif_final}, question:{closest}')
                                desired_csv = pd.read_csv('results/' + dif_final + ' results.csv')
                                break
                        if desired_csv.iloc[index]['Zero-Shot Evaluation'] == 1 and k > 0:
                            few_shot_input += desired_csv.iloc[index]['Codex Input']
                            few_shot_input += desired_csv.iloc[index]['Codex Output']+'\n\n'
                            k -= 1
                        if k == 0:
                            break
                    few_shot_input += results.iloc[i]['Codex Input']
                    start = time.time()
                    time.sleep(codex_time_delay) #to avoid an openai.error.RateLimitError
                    few_shot_output = openai.Completion.create(engine = codex_engine, 
                                                            prompt = few_shot_input, 
                                                            max_tokens = few_shot_max_tokens, 
                                                            temperature = engine_temperature, 
                                                            top_p = engine_topP)['choices'][0]['text']
                    print('Codex API call time: ' + str(time.time()-start) + '\n')

                #columns not properly labelled with 1's and 0's:
                else:
                    print('''A Question not labeled 1 for correct or 0 for incorrect was detected. 
                    You must go back and label all Codex Zero-Shot questions as correct or incorrect''')
                    raise ValueError

                results.loc[i, 'Few-Shot Input'] = few_shot_input
                results.loc[i, 'Few-Shot Output'] = few_shot_output
                results.to_csv(final_location, index=False)
        fewshot_embeddings += final_embeddings

def execute_GPT3_few_shot(finals):
    """
    Runs GPT-3 few-shot on questions_per questions for each course in courses.
    """
    fewshot_embeddings = [] # not sure about this yet
    for final in finals:
        final_location = 'results/' + final + ' results.csv'
        #initializing new columns in csv
        results = pd.read_csv(final_location)
        results['GPT-3 Few-Shot Input'] = ''
        results['GPT-3 Few-Shot Output'] = ''
        results['GPT-3 Few-Shot Evaluation'] = ''
        results.to_csv(final_location, index=False)

        #getting all qs and their embeddings
        final_qs = []
        final_embeddings = get_embeddings('data/embeddings/' + final + '_embeddings.json')
        for i in range(len(results['Original Question'])):
            question = results.loc[i, 'Original Question']
            final_qs.append(question)     
        if final != all_finals[0]: 
            #doing few shot for every question
            for i in range(len(final_qs)):
                k = few_shot_examples_desired

                #correct via zero-shot:
                if results.iloc[i]['GPT-3 Evaluation'] == 1:
                    print('no few shot needed for ' + final + ' question ' + str(i+1))
                    few_shot_input = 'n/a'
                    few_shot_output = 'n/a'

                #incorrect via zero-shot:
                elif 0 <= results.iloc[i]['GPT-3 Evaluation'] < 1:
                    few_shot_input = ''
                    print('doing GPT-3 few-shot for ' + final + ' question ' + str(i+1) + '...')

                    #to find the candidate questions information and use it if it works
                    for closest in get_most_similar(fewshot_embeddings, final_embeddings[i]):
                        for dif_final in finals:
                            embeddings = get_embeddings('data/embeddings/' + dif_final + '_embeddings.json')
                            if closest > len(embeddings):
                                closest -= len(embeddings)
                            else:
                                index = closest - 1
                                print(f'found closest question in {dif_final}, question:{closest}')
                                desired_csv = pd.read_csv('results/' + dif_final + ' results.csv')
                                break
                        if k > 0:
                            few_shot_input += 'Q: ' + desired_csv.iloc[index]['Original Question']
                            few_shot_input += '\nA:' + str(desired_csv.iloc[index]['Actual Solution']) + '\n\n'
                            k -= 1
                        if k == 0:
                            break
                    few_shot_input += 'Q: ' + results.iloc[i]['Original Question'] + '\nA:'
                    start = time.time()
                    time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
                    few_shot_output = openai.Completion.create(engine = gpt3_engine, 
                                                            prompt = few_shot_input, 
                                                            max_tokens = gpt3_max_tokens, 
                                                            temperature = engine_temperature, 
                                                            top_p = engine_topP)['choices'][0]['text']
                    print('GPT-3 API call time: ' + str(time.time()-start) + '\n')

                #columns not properly labelled with 1's and 0's:
                else:
                    print('''A Question not labeled 1 for correct or 0 for incorrect was detected. 
                    You must go back and label all GPT-3 Zero-Shot questions as correct or incorrect''')
                    raise ValueError

                results.loc[i, 'GPT-3 Few-Shot Input'] = few_shot_input
                results.loc[i, 'GPT-3 Few-Shot Output'] = few_shot_output
                results.to_csv(final_location, index=False)
        fewshot_embeddings += final_embeddings

def execute_GPT3_CoT_few_shot(finals):
    """
    Runs GPT3 CoT few-shot on questions_per questions for each course in courses.
    """
    fewshot_embeddings = [] # not sure about this yet
    for final in finals:
        final_location = 'results/' + final + ' results.csv'
        #initializing new columns in csv
        results = pd.read_csv(final_location)
        results['GPT-3 CoT Few-Shot Input'] = ''
        results['GPT-3 CoT Few-Shot Output'] = ''
        results['GPT-3 CoT Few-Shot Evaluation'] = ''
        results.to_csv(final_location, index=False)

        #getting all qs and their embeddings
        final_qs = []
        final_embeddings = get_embeddings('data/embeddings/' + final + '_embeddings.json')
        for i in range(len(results['Original Question'])):
            question = results.loc[i, 'Original Question']
            final_qs.append(question)     
        if final != all_finals[0]: 
            #doing few shot for every question
            for i in range(len(final_qs)):
                k = few_shot_examples_desired

                #correct via zero-shot:
                if results.iloc[i]['GPT-3 CoT Evaluation'] == 1:
                    print('no few shot needed for ' + final + ' question ' + str(i+1))
                    few_shot_input = 'n/a'
                    few_shot_output = 'n/a'

                #incorrect via zero-shot:
                elif 0 <= results.iloc[i]['GPT-3 CoT Evaluation'] < 1:
                    few_shot_input = ''
                    print('doing GPT-3 CoT few-shot for ' + final + ' question ' + str(i+1) + '...')

                    #to find the candidate questions information and use it if it works
                    for closest in get_most_similar(fewshot_embeddings, final_embeddings[i]):
                        for dif_final in finals:
                            embeddings = get_embeddings('data/embeddings/' + dif_final + '_embeddings.json')
                            if closest > len(embeddings):
                                closest -= len(embeddings)
                            else:
                                index = closest - 1
                                print(f'found closest question in {dif_final}, question:{closest}')
                                desired_csv = pd.read_csv('results/' + dif_final + ' results.csv')
                                break
                        if k > 0:
                            few_shot_input += 'Q: ' + desired_csv.iloc[index]['Original Question']
                            few_shot_input += '\nA:' + str(desired_csv.iloc[index]['Actual Solution']) + '\n\n'
                            k -= 1
                        if k == 0:
                            break
                    few_shot_input += 'Q: ' + results.iloc[i]['Original Question'] + '\nA: ' + CoT
                    start = time.time()
                    time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
                    few_shot_output = openai.Completion.create(engine = gpt3_engine, 
                                                            prompt = few_shot_input, 
                                                            max_tokens = gpt3_CoT_max_tokens, 
                                                            temperature = engine_temperature, 
                                                            top_p = engine_topP)['choices'][0]['text']
                    print('GPT-3 (for CoT) API call time: ' + str(time.time()-start) + '\n')

                #columns not properly labelled with 1's and 0's:
                else:
                    print('''A Question not labeled 1 for correct or 0 for incorrect was detected. 
                    You must go back and label all GPT-3 CoT Zero-Shot questions as correct or incorrect''')
                    raise ValueError

                results.loc[i, 'GPT-3 CoT Few-Shot Input'] = few_shot_input
                results.loc[i, 'GPT-3 CoT Few-Shot Output'] = few_shot_output
                results.to_csv(final_location, index=False)
        fewshot_embeddings += final_embeddings

if __name__ == "__main__":
    if args.Codex_Few_Shot:
        execute_few_shot(all_finals)
    if args.GPT3_Few_Shot:
        execute_GPT3_few_shot(all_finals)
    if args.GPT3_CoT_Few_Shot:
        execute_GPT3_CoT_few_shot(all_finals)