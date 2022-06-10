import openai
import pandas as pd
import random
import time

from embedding import get_embeddings, get_most_similar

finals = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']
topics = ['Regression', 'Classifiers', 'Logistic Regression', 'Features', 'Neural Networks', 'CNNs', 'MDPs', 'RNNs', 'Decision Trees']
gpt3_engine = "text-davinci-002"
engine_temperature = 0
engine_topP = 0
max_tokens = 200
gpt3_time_delay = 1

k = 3

#load all questions and their embeddings:
information = [] #stored in (question, embedding, topic) tuples.
for final in finals:
    sheet = pd.read_csv('data/csvs/' + final + '.csv')
    sheet = sheet.fillna('null')
    embeddings = get_embeddings('data/embeddings/' + final + '_embeddings.json')
    ind = 0
    for i in range(len(sheet['Question'])):
        if sheet.loc[i, "Question Number"] == 'null': #a null(empty entry) in question is treated as cutoff
            break
        if (sheet.loc[i, "Type"].lower() == "image"): 
            continue
        raw_question = sheet.loc[i, 'Question']
        q_topic = sheet.loc[i, 'Topic']
        print(f'final: {final}, len info:{len(information)},ind:{ind} ,len embeddings:{len(embeddings)}')
        information.append((raw_question, embeddings[ind], q_topic))
        ind += 1
      

#good from here up

for topic in topics:
    valid_qs = []
    for q in information:
        if q[2] == topic:
            valid_qs.append(q)
    #randomly pick k questions
    indicies = []
    while len(indicies)<k:
        ind = random.randint(0, len(valid_qs)-1)
        if ind not in indicies:
            indicies.append(ind)
    generate_input = ''
    for i, index in enumerate(indicies):
        generate_input += str(i+1) + '.' + valid_qs[i][0] + '\n' 
    generate_input += str(i+1) + '.'
    time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
    generated_question = openai.Completion.create(engine = gpt3_engine,
                                                  prompt = generate_input,
                                                  max_tokens = max_tokens,
                                                  temperature = engine_temperature,
                                                  top_p = engine_topP)['choices'][0]['text']

    
