import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

from ingestion import read_csv

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

def segregate_dataset(dataset):
    """
    Eliminate features not used
    segregate the dataset into X and y
    input: dataset to segregate
    output: X and y
    """

    # eliminate features not used for training
    features = ['lastmonth_activity','lastyear_activity','number_of_employees','exited']
    dataset = dataset[features]

    # data segregation
    predictors = features[:-1]
    target_variable = 'exited'
    X = dataset[predictors]
    y = dataset[target_variable]

    return X,y


#################Function for training the model
def train_model():
    """
    Train a logistic regression model for churn classification
    input: None
    output: trained model saved to disk
    """
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # import dataset
    filepath = os.path.join(dataset_csv_path, "finaldata.csv")
    dataset = read_csv(filepath)

    X,y = segregate_dataset(dataset)
    
    # fit the logistic regression to your data
    model.fit(X,y)
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    savingpath = os.path.join(model_path,'trainedmodel.pkl')
    with open(savingpath, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()