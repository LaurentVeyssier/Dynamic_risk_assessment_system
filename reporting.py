import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import json
import os
import ast

from diagnostics import (model_predictions, dataframe_summary, missing_data, 
                        execution_time, outdated_packages_list)
from ingestion import read_csv

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    
    # collect test dataset
    datasetpath = os.path.join(test_data_path, 'testdata.csv')
    dataset = read_csv(datasetpath)
    # perform prediction
    yhat = model_predictions(dataset)
    # calculate confusion matrix
    y = dataset['exited']
    cm = metrics.confusion_matrix(y, yhat)

    # Create cm plot
    f, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True,cmap='viridis', fmt='d', linewidths=.5, annot_kws={"fontsize":15})
    plt.xlabel('Predicted Class', fontsize = 15)
    ax.xaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.ylabel('True Class', fontsize = 15)
    ax.yaxis.set_ticklabels(['Not Churned', 'Churned'])
    plt.title('Confusion matrix', fontsize = 20)
    
    # write the confusion matrix to the workspace
    savepath = os.path.join(model_path,'confusionmatrix.png')
    f.savefig(savepath)



    # Additional Statistics
    # compute classification report
    cr = metrics.classification_report(y, yhat, output_dict=True)
    # Collect statistics
    statistics = dataframe_summary()
    missingdata = missing_data()
    timings = execution_time()
    dependencies = outdated_packages_list()
    # collect ingested files
    filepath = os.path.join(dataset_csv_path,'ingestedfiles.txt')
    with open(filepath, 'r') as f:
        ingestedfiles = ast.literal_eval(f.read())

    # Produce pdf report
    # 1- list of ingested files
    ingestedfiles = pd.DataFrame(ingestedfiles, columns=['Ingested files'])
    col_names = ingestedfiles.columns.tolist()
    data = ingestedfiles.values
    rowLabels = ingestedfiles.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(3,3))
    plt.title('Ingested files', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right', rowLabels=rowLabels)
    plt.tight_layout()

    # 2- summary statistics
    col_names = ['lastmonth_activity','lastyear_activity','number_of_employees','exited']
    data = np.array(statistics).reshape(3,4)
    # Plot table
    fig, ax = plt.subplots(1, figsize=(10,2))
    plt.title('Summary statistics', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=['mean','median','std'])

    # 3- Confusion matrix
    # Already created above

    # 4- classification report
    df = pd.DataFrame(cr).transpose()
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(10,5))
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.title('Classification report', fontsize = 20)
    plt.tight_layout()

    # 5- Missing data
    df = pd.DataFrame(data=missingdata, index = dataset.columns.tolist(), columns=['missing data'])
    col_names = df.columns.tolist()
    data = df.values
    rowLabels = df.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4,6))
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.title('Missing data', fontsize = 20)
    plt.tight_layout()

    # 6- Timing of execution
    timing = pd.DataFrame(timings, columns=['Duration (sec)'])
    col_names = timing.columns.tolist()
    data = timing.values
    rowLabels = ["Ingestion step", 'Training step']
    # Plot table
    fig, ax = plt.subplots(1, figsize=(4,4))
    plt.title('Execution time', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right', rowLabels=rowLabels)
    plt.tight_layout()

    # 7- dependencies status
    col_names = dependencies.columns.tolist()
    data = dependencies.values
    rowLabels = dependencies.index.tolist()
    # Plot table
    fig, ax = plt.subplots(1, figsize=(5,5))
    plt.title('dependencies status', fontsize = 20)
    ax.axis('off')
    table = plt.table(cellText=data, colLabels=col_names, loc='center',colLoc='right',rowLabels=rowLabels)
    plt.tight_layout()



    def save_multi_image(filename):
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()

    filename = "report.pdf"
    save_multi_image(filename)



if __name__ == '__main__':
    score_model()
