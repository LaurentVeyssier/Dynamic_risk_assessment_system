from flask import Flask, session, jsonify, request
import pandas as pd

import diagnostics, scoring, ingestion
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Scoring Endpoint
@app.route("/")
def greetings():        
    #welcoming message
    return 'Welcome to our model API'

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    if request.method == 'POST':
        file = request.files['filename']
        dataset = ingestion.read_csv(file)
        return diagnostics.model_predictions(dataset)
    if request.method == 'GET':
        file = request.args.get('filename')
        dataset = ingestion.read_csv(file)
        return {'predictions': str(diagnostics.model_predictions(dataset))}

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():        
    #check the score of the deployed model
    return {'F1 score': scoring.score_model()}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_stats():        
    #check means, medians, and modes for each column
    return {'key statistics': {c:{'mean':diagnostics.dataframe_summary()[i],
                                  'median':diagnostics.dataframe_summary()[i+4],
                                  'std':diagnostics.dataframe_summary()[i+8]} 
    for c,i in zip(['lastmonth_activity','lastyear_activity','number_of_employees'], range(3))
                                }
            }

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():        
    #check timing and percent NA values
    missing_data = diagnostics.missing_data()
    timing = diagnostics.execution_time()
    dependency_check = diagnostics.outdated_packages_list()
    return {'execution time': {step:duration 
                for step, duration in zip(['ingestion step','training step'],
                                            timing)}, 
            'missing data': {col:pct 
                for col, pct in zip(['lastmonth_activity',
                                    'lastyear_activity',
                                    'number_of_employees',
                                    'exited'], missing_data)},
            'dependency check':[{'Module':row[0], 
                                'Version':row[1][0], 
                                'Vlatest':row[1][1]} 
                                for row in dependency_check.iterrows()]
            }

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
