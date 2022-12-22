import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from io import StringIO
import subprocess

from ingestion import read_csv
from training import segregate_dataset

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##################Function to get model predictions
def model_predictions(dataset=None):
    """
    read the deployed model and a test dataset, calculate predictions
    input (optional): dataset to use for prediction evaluation
    output: list of predictions from deployed model
    """
    # if not dataset provided then use test dataset
    if dataset is None:
        datasetpath = os.path.join(test_data_path, 'testdata.csv')
        dataset = read_csv(datasetpath)

    # collect deployed model
    modelpath = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    # segregate test dataset
    X, y = segregate_dataset(dataset)

    # evaluate model on test set
    yhat = model.predict(X)

    return yhat


##################Function to get summary statistics
def dataframe_summary():
    """calculate summary statistics on the dataset columns"""

    # collect dataset
    datasetpath = os.path.join(dataset_csv_path, 'finaldata.csv')
    dataset = read_csv(datasetpath)

    # Select numeric columns
    numeric_col_index = np.where(dataset.dtypes != object)[0]
    numeric_col = dataset.columns[numeric_col_index].tolist()

    # compute statistics per numeric column
    means = dataset[numeric_col].mean(axis=0).tolist()
    medians = dataset[numeric_col].median(axis=0).tolist()
    stddevs = dataset[numeric_col].std(axis=0).tolist()

    statistics = means
    statistics.extend(medians)
    statistics.extend(stddevs)

    return statistics


##################Function to get missing data
def missing_data():
    """calculate missing data on the dataset
    return % of missing data per column
    """

    # collect dataset
    datasetpath = os.path.join(dataset_csv_path, 'finaldata.csv')
    dataset = read_csv(datasetpath)

    # compute missing data % per column
    missing_data = dataset.isna().sum(axis=0)
    missing_data /= len(dataset) *100

    return missing_data.tolist()


##################Function to get timings
def execution_time():
    """calculate timing of training.py and ingestion.py"""
    timing_measures = []

    # timing ingestion step
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    end_time = timeit.default_timer()
    duration_step = end_time - start_time
    timing_measures.append(duration_step)

    # timing ingestion step
    start_time = timeit.default_timer()
    os.system('python training.py')
    end_time = timeit.default_timer()
    duration_step = end_time - start_time
    timing_measures.append(duration_step)

    return timing_measures


def execute_cmd(cmd):
    """execute a pip list type cmd
    input: pip list type of cmd
    return: output of cmd in dataframe format
    """
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # text=True then no need to decode bytes to str
    b = StringIO(a.communicate()[0].decode('utf-8'))
    df = pd.read_csv(b, sep="\s+")
    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')
    return df

##################Function to check dependencies
def outdated_packages_list():
    """get a list of dependencies and versions
    input: None
    output: dataframe with list of outdated dependencies, 
            version as per requirements.txt file,
            and latest version available
    """

    # collect outdated dependencies (for current virtual env)
    cmd = ['pip', 'list', '--outdated']
    df = execute_cmd(cmd)
    df.drop(['Version','Type'], axis=1, inplace=True)

    # collect all dependencies (for current virtual env)
    cmd = ['pip', 'list']
    df1 = execute_cmd(cmd)
    df1 = df1.rename(columns = {'Version':'Latest'})

    # collect dependencies as per requirements.txt file
    requirements = pd.read_csv('requirements.txt', sep='==', header=None, names=['Package','Version'], engine='python')
    requirements = requirements.set_index('Package')

    # assemble target and latest versions for requirements.txt dependencies
    dependencies = requirements.join(df1)
    for p in df.index:
        if p in dependencies.index:
            dependencies.at[p, 'Latest'] = df.at[p,'Latest']
    
    # keep only outdated dependencies (ie latest version exists)
    dependencies.dropna(inplace=True)

    return dependencies


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
