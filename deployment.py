import os
import json
import shutil

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle():
    
    #copy the latest pickle file and its latestscore.txt file into the deployment directory
    for file in ['latestscore.txt', 'trainedmodel.pkl']:
        shutil.copy(os.path.join(model_path,file),
                    os.path.join(prod_deployment_path,file))
        
    #copy the ingestfiles.txt file into the deployment directory
    shutil.copy(os.path.join(dataset_csv_path,'ingestedfiles.txt'),
                    os.path.join(prod_deployment_path,'ingestedfiles.txt'))

if __name__ == '__main__':
    store_model_into_pickle()