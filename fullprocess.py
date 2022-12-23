
import json
import os
import ast
from sklearn import metrics
import logging

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

# Initialize logging
logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']

move_to_next_step = False
logging.info("Launching automated monitoring")
##################Check and read new data
#first, read ingestedfiles.txt
filepath = os.path.join(prod_deployment_path,'ingestedfiles.txt')
with open(filepath, 'r') as f:
    ingestedfiles = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files = os.listdir(input_folder_path)
files = [file for file in files if file not in ingestedfiles]

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if files !=[]:
    logging.info("ingesting new files")
    ingestion.merge_multiple_dataframe()
    move_to_next_step = True
else:
    logging.info("No new files - ending process")
##################Checking for model drift
"""There are two possible scenarios here:
- We train a new model on new data and then compare the performance of the new model 
and the existing model on the test data we set aside before.
- We evaluate our existing model on the new data we have. If its performance falls 
below its recorded performance on test data, we train a new model on the new data and deploy it.

In the first case, we cannot explicitly say that the existing model drifted. 
The model trained on new data simply performed better when evaluated on the same test data. 
Whereas in the second scenario, the performance of our existing model degraded 
on new data prompting us to train another model on the new data."""

"""You can manually change the dataset instead the latest score and 
get an worse result only to make sure your code runs fine."""

#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
if move_to_next_step:
    scorespath = os.path.join(prod_deployment_path, 'latestscore.txt')
    with open(scorespath, 'r') as f:
         latest_score = ast.literal_eval(f.read())
    
    filepath = os.path.join(output_folder_path,'finaldata.csv')
    dataset = ingestion.read_csv(filepath)
    new_yhat = diagnostics.model_predictions(dataset)
    new_y = ingestion.read_csv(filepath)['exited']
    new_score = metrics.f1_score(new_y, new_yhat)
    logging.info(f'latest score: {latest_score}, new score: {new_score}')

    if new_score >= latest_score:
        move_to_next_step = False  # No model drift, keep existing model
        logging.info('No model drift - ending process')

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if move_to_next_step:  # model drift, move to retraining
    logging.info('training new model')
    training.train_model()
    scoring.score_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if move_to_next_step:
    logging.info('deploying new model')
    deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
if move_to_next_step:
    logging.info('producing reporting and calling apis for statistics')
    reporting.score_model()
    os.system('python apicalls.py')






