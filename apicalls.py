import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 

filepath = os.path.join(test_data_path,'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.get(URL + '/prediction' + f'?filename={filepath}').content
response2 = requests.get(URL + '/scoring').content
response3 = requests.get(URL + '/summarystats').content
response4 = requests.get(URL + '/diagnostics').content

#combine all API responses
responses = {'Predictions':response1.decode('utf-8'),
            'Scoring':response2.decode('utf-8'),
            'Statistics':response3.decode('utf-8'),
            'Diagnostics':response4.decode('utf-8')}

#write the responses to your workspace
filepath = os.path.join(model_path,'apireturns.txt')
with open(filepath,'w') as f:
    f.write(json.dumps(responses))


