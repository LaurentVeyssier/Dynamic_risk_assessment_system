import pickle
import os
from sklearn import metrics
import json

from ingestion import read_csv
from training import segregate_dataset

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 

# import test dataset
filepath = os.path.join(test_data_path, 'testdata.csv')
testdata = read_csv(filepath)

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    # load trained model
    modelpath = os.path.join(model_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    # segregate test dataset
    X, y = segregate_dataset(testdata)

    # evaluate model on test set
    yhat = model.predict(X)
    score = metrics.f1_score(y, yhat)

    # save as latest score
    scorespath = os.path.join(model_path, 'latestscore.txt')
    with open(scorespath, 'w') as f:
         f.write(str(score))

    return score


if __name__ == '__main__':
    score_model()