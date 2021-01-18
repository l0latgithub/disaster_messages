# disaster_messages app

## Table of Content
  1. [Descriptions](#descriptions)
  2. [Requirements](#requirements)
  3. [Instructions](#instructions)
  4. [Results](#results)
  5. [Deployment](#deployment)
  5. [Acknowledgement](#acknowledgement)

### Descriptions
  A web app was built with Natural Language Process(NLP) model with pipeline to classify disaster messages or tweets. Emergency response team could use this tool to determine how to respond during an disaster event.

### Requirements
  
  The files/modules used to build this app could be refered as follows.
  
  python -m pip install -r requirements.txt

### Instructions
  1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
  2. Run the following command to run the web app.
  
    `python app/run.py`

### Results
  This web app is deployed at [Heroku](https://messageapp2021.herokuapp.com/)

### Deployment
  Heroku web app deployment could be refered [here](https://devcenter.heroku.com/articles/git)

### Acknowledgement
  1. [Udacity](https://www.udacity.com/) for providing the framework.
  2. [Figure Eight](https://appen.com//) for providing the relevant dataset to train the model
