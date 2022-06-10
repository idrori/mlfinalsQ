import openai
import json
from sentence_transformers import util
import os
import pandas as pd

openai.api_key = os.getenv('OpenAI_API_Key')
finals = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']
embedding_engine = 'text-similarity-babbage-001'

def make_embeddings(embedding_engine, final):
    """
    Embeds one final.
    """
    list_of_embeddings = []
    print("Currently embedding " + final + "...")
    sheet = pd.read_csv('data/csvs/' + final + '.csv')
    sheet = sheet.fillna('null')
    for i in range(len(sheet['Question'])):
        if sheet.loc[i, "Question Number"] == 'null': #a null(empty entry) in question is treated as cutoff
            break
        if (sheet.loc[i, "Type"].lower() == "image"):  
            continue
        raw_question = sheet.loc[i, 'Question']
        embedding = openai.Embedding.create(input = raw_question, 
                                            engine = embedding_engine)['data'][0]['embedding']
        list_of_embeddings.append(embedding)
    embeddings = {'list_of_embeddings':list_of_embeddings}
    if not os.path.isdir('data'):
        os.mkdir('data')
    folder_path = 'data/embeddings'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    with open('data/embeddings/' + final + '_embeddings.json', 'w') as f:
        f.write(json.dumps(embeddings))

def get_embeddings(embeddings_file):
    """
    Retrieves embeddings from embeddings_file. Embeddings are assumed to be (n x d).
    """
    with open(embeddings_file, 'r') as f:
        points = json.load(f)['list_of_embeddings']
    return points

def get_most_similar(embeddings, target_embedding):
    """
    Returns most similar questions, while they are in their embedded form, 
        to the target, target_embedding, via cosine similarity.
    """
    cos_sims = []
    cos_to_num = {}
    for j in range(len(embeddings)):
        cos_sim = util.cos_sim(target_embedding, embeddings[j]).item()
        cos_to_num[cos_sim] = j
        cos_sims.append(cos_sim)
    ordered = sorted(cos_sims, reverse=True)
    closest_qs = [cos_to_num[val]+1 for val in ordered]
    return closest_qs

if __name__ == "__main__":
    for final in finals:
        if not os.path.exists('data/embeddings/' + final + '_embeddings.json'):
            make_embeddings(embedding_engine, final)