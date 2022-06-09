import os
import openai 
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
# if an argument is passed in as True, we do it
parser.add_argument("--Codex")
parser.add_argument("--GPT3")
parser.add_argument("--GPT3_CoT")
args = parser.parse_args()

column_labels = ['Question', 'Original Question', 'Actual Solution']
if args.Codex == 'True':
    column_labels += ['Codex Input', 'Codex Output', 'Zero-Shot Evaluation']
if args.GPT3 == 'True':
    column_labels += ['GPT-3 Output', 'GPT-3 Evaluation']
if args.GPT3_CoT == 'True':
    column_labels += ['GPT-3 CoT Input', 'GPT-3 CoT Output', 'GPT-3 CoT Evaluation']

openai.api_key = os.getenv('OpenAI_API_Key')

finals_to_zero_shot = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']
codex_engine = "code-davinci-002"
gpt3_engine = "text-davinci-002"
engine_temperature = 0
engine_topP = 0
zero_shot_max_tokens = 1024
gpt3_max_tokens = 200
gpt3_CoT_max_tokens = 1000
codex_time_delay = 3
gpt3_time_delay = 1

# for prompt formatting:
docstring_front = '''"""\n''' 
docstring_back = '''\n"""\n'''
context_array = ['write a program', 'using sympy', 'using simulations']
prompt_prefix = 'that answers the following question:'
explanation_suffix = "\n\n'''\nHere's what the above code is doing:\n1."
CoT = "Let's think step by step."

def execute_zero_shot(finals):
    """
    Runs zero-shot on each final in finals. 
    An individual CSV file of the results is made for each final in finals.
    The embeddings for all of the questions for all of the courses in courses are located in embeddings_location.
    """
    for final in finals:
        questions = []
        answers = []
        indicators = []
        sheet = pd.read_csv('finals/' + final + '.csv')
        sheet = sheet.fillna('null')
        for i in range(len(sheet['Question'])):
            if sheet.loc[i, "Question Number"] == 'null': #a null(empty entry) in question is treated as cutoff
                break
            if (sheet.loc[i, "Part"].lower() == "image"): 
                continue
            raw_question = sheet.loc[i, 'Question']
            indicator = str(sheet.loc[i, 'Question Number']) + str(sheet.loc[i, 'Part'])
            answer_to_question = sheet.loc[i, 'Solution']
            indicators.append(indicator)
            questions.append(raw_question)
            answers.append(answer_to_question)
        
        rows = []
        for i, question_indicator in enumerate(indicators):
            original_question = questions[i]
            question_answer = answers[i]
            row = [question_indicator, original_question, question_answer]
            print('Running Zero-Shot on ' + final + ' question ' + question_indicator + '...')
            start = time.time()

            if args.Codex == 'True':
                time.sleep(codex_time_delay) #to avoid an openai.error.RateLimitError
                codex_input = docstring_front + context_array[0] + ' ' + prompt_prefix + ' ' + questions[i] + docstring_back
                codex_output = openai.Completion.create(engine = codex_engine, 
                                                        prompt = codex_input, 
                                                        max_tokens = zero_shot_max_tokens, 
                                                        temperature = engine_temperature, 
                                                        top_p = engine_topP)['choices'][0]['text']
                row += [codex_input, codex_output, '']

            if args.GPT3 == 'True':
                time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
                gpt3_output = openai.Completion.create(engine = gpt3_engine, 
                                                    prompt = original_question, 
                                                    max_tokens = gpt3_max_tokens, 
                                                    temperature = engine_temperature, 
                                                    top_p = engine_topP)['choices'][0]['text']
                row += [gpt3_output, '']

            if args.GPT3_CoT == 'True':
                time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
                gpt3_CoT_input = 'Q: ' + original_question + "\nA: " + CoT
                gpt3_CoT_output = openai.Completion.create(engine = gpt3_engine,
                                                    prompt = gpt3_CoT_input,
                                                    max_tokens = gpt3_CoT_max_tokens,
                                                    temperature = engine_temperature,
                                                    top_p = engine_topP)['choices'][0]['text']
                row += [gpt3_CoT_input, gpt3_CoT_output, '']
                
            end = time.time()
            print('API call time: ' + str(end-start) + '\n')
            rows.append(row)
        info = pd.DataFrame(rows, columns=column_labels)
        folder_path = 'results'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        final_results_location = 'results/' + final + ' results.csv'
        info.to_csv(final_results_location, index=False)

if __name__ == "__main__":
    execute_zero_shot(finals_to_zero_shot)