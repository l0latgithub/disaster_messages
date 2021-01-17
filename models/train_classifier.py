import sys
import pandas as pd
import pickle
import sqlite3
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from statistics import mean

def load_data(database_filepath):
    
    # Load data from sqlite database
    engine = create_engine( 'sqlite:///{}'.format(database_filepath) )
    df = pd.read_sql_table('rawdata',con=engine)
    
    # input are only messages
    X = df['message']
    
    # Since id, message, orginal, genre are not not labels, they are discarded
    Y = df.drop(labels=['id','message','original','genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    import re
    # Use regex to find all urls and replace them to be a constant string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Use NLTK word tokenizer and Lemmatizer to tokenize and lemmatize the messges
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Tokens are normalized to lower case and remove leading/trailing spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    A customerized transformer to detect leading verb for the messages
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    
    """
    Build pipeline model with tfidf transformer
    
    Since the labels are multilabels, multioutputclassifer is used
    
    k-nearest neighbors classifer is chosen for this study
    
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

#         ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    def avg_score(y_true,y_pred):
        
        """
        This is a multilabels problem
        Average accuracy, precision, recall were calculated to quickly quantify
        model performace.
        """
        
        # prediction output is numpy array and labels are dataframe
        # labels are converted to numpy array for easy calculation
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
        
        print (y_pred.shape, y_true.shape)
        
        # lists for precision, recall, and accuracy storage
        avg_precision = []
        avg_recall = []
        avg_accuracy = []

        for col in range(y_true.shape[1]):

            true = y_true[:,col]
            pred =  y_pred[:, col]

            precisions, recalls, fscores, supports =  precision_recall_fscore_support( true,  pred )
            accuracy = accuracy_score(true,  pred )
            
            avg_precision.append( sum(precisions*supports)/sum(supports) )
            avg_recall.append( sum(recalls*supports)/sum(supports) )
            avg_accuracy.append(accuracy)

        return mean(avg_accuracy), mean(avg_precision), mean(avg_recall)
    
    print ("Start prediction")
    y_pred = model.predict(X_test)
    print ("end of prediction")
    avg_accuracy, avg_precision, avg_recall = avg_score(Y_test, y_pred)
    
    print ("model average accuracy", avg_accuracy)
    print ("model average precision", avg_precision)
    print ("model average recall", avg_recall)


def save_model(model, model_filepath):
    # save model to pickle file
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        # GridSearchCV was used to get the parameters
        # parameters = {
        #    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #    'features__text_pipeline__vect__max_df': (0.5, 1.0),
        #    'features__text_pipeline__vect__max_features': (None, 3000),
        #    'features__transformer_weights': ({'text_pipeline': 1, 'starting_verb': 0.5},{'text_pipeline': 1, 'starting_verb': 0.2}),
        #    'clf__estimator__n_estimators': [50, 200],
        #    'clf__estimator__learning_rate': [1,0.3],}
        # cv = GridSearchCV(model, cv=5, param_grid = parameters)

        params_gcv = {'clf__estimator__learning_rate': 0.3,
        'clf__estimator__n_estimators': 200,
        'features__text_pipeline__vect__max_df': 1.0,
        'features__text_pipeline__vect__ngram_range': (1, 1),
        'features__transformer_weights': {'text_pipeline': 1, 'starting_verb': 0.5}}

        model.set_params(
            **params_gcv
        )
    
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()