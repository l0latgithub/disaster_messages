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
  
      │   LICENSE.md                      # MIT license
      │   message.py                      # import the app
      │   nltk.txt                        # how to use NLTK on Heroku https://devcenter.heroku.com/articles/python-nltk
      │   Procfile                        # Heroku apps include a Procfile that specifies the commands that are executed by the app on startup
      │   README.md
      │   requirements.txt                # All modules used in this app
      │   util.py                         # library used in this app
      │
      ├───app
      │   │   run.py                      # Python program to render picture and return classification results
      │   │   __init__.py                 # python program to return the app object
      │   │
      │   └───templates
      │           go.html                 # classification result html template
      │           master.html             # web app master html
      │
      ├───data
      │       DisasterResponse.db         # SQLite database of processed message and label data
      │       disaster_categories.csv     # Raw label data
      │       disaster_messages.csv       # Raw message data
      │       process_data.py             # Data extraction, transform, and load (ETL) python program
      │
      └───models
              classifier.pkl              # NLP classification model saved
              train_classifier.py         # NLP model training program
              util.py                     # library used in this app

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
     To run the program locally, has to modify app/run.py accordingly.
     
     `python app/run.py`
     
     a. Run locally: uncomment line 31 and 143
     
     b. Deploy on Herok: comment line 31 and 143, uncomment 141

### Results
  This web app is deployed at [Heroku](https://messageapp2021.herokuapp.com/)

### Deployment
  Heroku web app deployment could be refered [here](https://devcenter.heroku.com/articles/git)

### Acknowledgement
  1. [Udacity](https://www.udacity.com/) for providing the framework.
  2. [Figure Eight](https://appen.com//) for providing the relevant dataset to train the model
