# Disaster Response Pipeline Project
### Summary
This is a website that displays statistics about tweets during a disaster, and predicts what type of disaster tweet a user inputted message is.
This helps people by allowing organizations to identify key areas in need of assistance during a disaster.
### Files
```
│   .gitignore : files ignored by git
│   README.md : This file
│
├───app
│   │   run.py : Python file that when run launches the web app
│   │
│   └───templates
│           go.html : template file for page where user enters input
│           master.html : template file for main page
│
├───data
│       DisasterResponse.db : SQLite file containing the table dataframe
│       disaster_categories.csv : csv file with the categories of disasters
│       disaster_messages.csv : csv file with associated messages for disaster
│       process_data.py  : Python file that processes the csv file to create DisasterResponse.db
│
├───models
│       train_classifier.py : Python file that creates model in classifier.pkl
        classifier.pkl (ignored) : File containing pickled xgboost model so it can be reused 
│
├───notebooks
│       ETL Pipeline Preparation.ipynb : Jupyter notebook containing how I explored and executed the ETL pipeline
│       ML Pipeline Preparation.ipynb : Jupyter notebook containing how I explored and executed the ML pipeline
│
└───screenshots
        app_running.png : Screenshot of the web app running in the terminal
        classifier_input.png : Screenshot of the input to the go page
        classifier_output.png : Screen of shot of the output on the go page
        db_visualizations.png : Visualizations on the homepage from the DB
        succesful_model_training.png : Screenshot of the terminal showing that the model was trained successfully
        successful_data_loading.png : Screenshot of the terminal showing that the data was loaded successfully
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

This was done as a part of the Udacity Data Science Nanodegree
