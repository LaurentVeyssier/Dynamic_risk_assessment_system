import pandas as pd
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# get current directory
working_dir = os.getcwd()
# get current date
today = datetime.today().strftime('%Y%m%d')

def read_csv(filename):
    """ read a csv file using filename
    input: filenames to read
    output: dataframe
    """
    return pd.read_csv(filename)

#############Function for data ingestion
def merge_multiple_dataframe():
    """ 
    combine multiple datasets into one master file
    record ingested files and save to disk
    input: None
    output: Master dataset and list of ingested files saved to disk
    """

    # master dataset placeholder
    finaldata = pd.DataFrame()
    # ingested files placeholder
    ingestedfiles = []

    #check for datasets
    files = os.listdir(input_folder_path)

    # compile datasets together and store ingested file names
    for file in files:
        filepath = os.path.join(input_folder_path,file)
        temp = read_csv(filepath)
        finaldata = pd.concat([finaldata,temp], axis=0)
        ingestedfiles.append(file)

    # drop duplicates
    finaldata.drop_duplicates(inplace=True)

    # write dataset to an output master file
    savepath = os.path.join(output_folder_path,'finaldata.csv')
    finaldata.to_csv(savepath, index=False)

    # save ingested files with timestamp
    savepath = os.path.join(output_folder_path,'ingestedfiles.txt')      # f'{today}_ingestedfiles.txt'
    with open(savepath, 'w') as f:
        f.write(str(ingestedfiles))

    

if __name__ == '__main__':
    merge_multiple_dataframe()
