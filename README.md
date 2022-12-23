# Dynamic risk assessment system
Project #4 from Udacity's ML DevOps engineer Nanodegree 

# Project Overview: Training, Scoring, and Deploying an ML Model
This is the final project of Udacity ML DevOps Engineer Nanodegree. The project covers the full scope of ML model deployment with continuous ingestion of data and detection of model drift leading to retraining / redeployment decisions.
The model is made accessible through an API with multiple endpoints prodiving inference and various statistics about the model performance, the data and overall pipeline health.

# Context
The project aims at building a classifier for the prediction of customer churn and overall reisk assessment.
Synthetic datasets are provided to train and simulate continuous operations. The emphasis is put on the MLOps components rather than the quality of the model.

# Workflow overview
The workflow is broken down into the following components which are all separated:
- data ingestion
- model training (logisticRegression sklearn model for binary classification)
- model scoring
- deployment of pipeline artifacts into production
- model monitoring, reporting and statistics - API set-up for ML diagnostics and results
- process automation with data ingestion and model drift detection using CRON job
- retraining / redeployment in case of model drift

# How to use
The project is deployed under windows python WSL2 linux. Deployed API using Flask framework.
The model API should be launched before executing project components. This can be achieved by running app.py script instantiating multiple project API endpoints including inference capability. Other components of the project include:
- ingestion.py to ingest data and prepare model training
- training.py to train a logisticregression model
- scoring.py to score the model in production against a test dataset
- deployment.py to deploy key artifacts to production (in particular the trained model artifact)
- diagnostics.py gather various analysis and diagnostics
- apicalls.py calls all diagnostics through the API and generate a consolidated report
- reporting.py allows to generate a full pdf report gathering performance plots, metrics and other useful statistics
- fullprocess.py should be run regularly using a CRON job. It monitors new data availability, checks model drift, decides to retrain and redeploy an updated model in case shifting is detected.

# Initialization
For the process implementation, a first model should be trained and implemented. this can be done by running all individual components (ingestion, training, scoring, deployment) using practicedata and practicemodels folders in the config.json file (modify input_folder, output_model variables). Once initialization is performed, modify config.json back to production stage using sourcedata and models folders respectively, then implement fullprocess.py script for process automation in production.

# Cron job implementation
- activate cron jobs in WSL2 using sudo service cron start (if not already active)
- create a new cron job using crontab -e
- the cron job should run the fullprocess.py script every 10 minutes in order to automate the whole process from data ingestion to model redeployment as needed



