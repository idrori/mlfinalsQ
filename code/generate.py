import openai
import pandas as pd
import random
import time
import os
from embedding import get_embeddings, get_most_similar

openai.api_key = os.getenv('OpenAI_API_Key')
finals = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']
topics = ['Regression', 'Classifiers', 'Logistic Regression', 'Features', 'Neural Networks', 'CNNs', 'MDPs', 'RNNs', 'Decision Trees']
gpt3_engine = "text-davinci-002"
embedding_engine = 'text-similarity-babbage-001'
engine_temperature = 0
engine_topP = 0
max_tokens = 1000
gpt3_time_delay = 1
k = 3

#load all questions and their embeddings:
information = [] #stored in (question, embedding, topic, question number, final) tuples.
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
        information.append((raw_question, embeddings[ind], q_topic, str(sheet.loc[i, "Question Number"])+str(sheet.loc[i, "Part"]), final))
        ind += 1

#THIS ROUTINE GENERATES A NEW QUESTION FOR EACH TOPIC
for topic in topics:
    print(f'\n\n\n------------{topic}--------------\n\n\n')
    valid_qs = []
    for q in information:
        if q[2] == topic:
            valid_qs.append(q)

    #randomly pick k questions
    indicies = random.sample(range(0, len(valid_qs)), k)

    #make input string
    generate_input = ''
    for i, index in enumerate(indicies):
        generate_input += str(i+1) + '. ' + valid_qs[index][0] + '\n' 
    generate_input += str(i+2) + '. '
    print(f'This is the input:\n {generate_input}')

    #generate new question
    time.sleep(gpt3_time_delay) #to avoid an openai.error.RateLimitError
    generated_question = openai.Completion.create(engine = gpt3_engine,
                                                  prompt = generate_input,
                                                  max_tokens = max_tokens,
                                                  temperature = engine_temperature,
                                                  top_p = engine_topP)['choices'][0]['text']
    print(f'This is the generated question:\n{generated_question}\n')
    gen_emb = openai.Embedding.create(input = generated_question, 
                                      engine = embedding_engine)['data'][0]['embedding']

    #find most similar real question:
    embs = [inf[1] for inf in information]
    sim_index = get_most_similar(embs, gen_emb)[0] - 1
    print(f'most similar question to generated question: {information[sim_index][4]} question {information[sim_index][3]}')
    
