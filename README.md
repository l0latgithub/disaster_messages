# Disaster Response Pipeline Project

## Table of Contents
    1. [Description](#description)
    [Installation](#description)
    2. [Instructions](#instructions)
    3. [Acknowledgement](#acknowledgement)

### Descriptions <a name="description"></a>
This project aims to build a Natural Language Processing (NLP) web app to categorize messages. The project has three
major components
    1. Build sklearn ETYL pipeline to process data, clean, and save data to a SQLite database.
    2. Develop a NLP processing model to classify the text messages.
    3. Deploy the model as a web app to show the results.
### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`
### Acknowledgement <a name="acknowledgement"></a>
1. [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
2. [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model
