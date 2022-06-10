import json
import pandas as pd
import os 

question  =  {'Question number': None,
              'Sub-Question number': None,
              'Question' : None,
              'Solution' : None}

FINALS = ['Fall_2017', 'Spring_2018', 'Fall_2018', 'Spring_2019', 'Fall_2019', 'Spring_2021', 'Fall_2021', 'Spring_2022']

def get_file_q_num(n):
    """
    returns 2-digit string representing a given number n
    """
    if n < 10:
        return "0" + str(n)
    else:
        return str(n)

if __name__ == "__main__":
    if not os.path.isdir('data'):
        os.mkdir('data')
    folder_path = 'data/jsons'
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    for final in FINALS:
        final_path = 'data/jsons/' + final
        if not os.path.isdir(final_path):
            os.mkdir(final_path)
        sheet = pd.read_csv('data/csvs/' + final + '.csv')
        sheet = sheet.fillna('null')

        for i in range(len(sheet['Question'])):
            new_json = question
            if sheet.loc[i, "Question Number"] == 'null': #a null(empty entry) in question is treated as cutoff
                break
            if (sheet.loc[i, "Type"].lower() == "image"): 
                continue
            new_json['Question'] = sheet.loc[i, 'Question']
            new_json['Solution'] = sheet.loc[i, 'Solution']
            new_json['Question number'] = str(sheet.loc[i, 'Question Number'])
            new_json['Sub-Question number'] = sheet.loc[i, 'Part']
            json_object = json.dumps(new_json, indent = 7)
            fname = 'data/jsons/' + final + '/' + final + "_Question_" + get_file_q_num(int(sheet.loc[i, 'Question #'])) + "_" + sheet.loc[i, 'Sub-Question #'] + ".json"
            with open(fname, "w") as outfile:
                outfile.write(json_object)