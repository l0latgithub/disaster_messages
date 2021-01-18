from app import app
import json
import plotly
import pandas as pd


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals.joblib import joblib
# import sklearn.external.joblib as extjoblib
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from wordcloud import WordCloud

from plotly.graph_objs import Scatter
from plotly.offline import plot
import random

from util import Tokenizer,StartingVerbExtractor

# from util import tokenize
# Need the following line to run local?
# app = Flask(__name__)

# import os.path
# for filedir,_, filename in os.walk('.'):
#     print (filedir, filename)

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('rawdata', engine)

# # load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    """
    Web app index page. It renders two pictures about the data. One bar chart and word cloud.
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Plot Message cloud, this part need optimization and it is very slow at this moment
    allwords = " ".join( df['message'] )
    wordcloud = WordCloud().generate(allwords)
    alltexts = list( wordcloud.words_.keys() )[:30]
    num_words = 30
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(num_words)]
    weights = [int(num*60) for num in ( list( wordcloud.words_.values() )[:30] ) ]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'width': '500',
                'height': '600',
                'automargin': 'True',
                'titlefont': {'size': '30'}
            }
        },
        
        {
            'data': [
                Scatter(
                    x=[random.random() for i in range(num_words)],
                    y=[random.random() for i in range(num_words)],
                    mode='text',
                    text=alltexts,
                    marker={'opacity': 0.3},
                    textfont={'size': weights,
                           'color': colors}
                )
            ],

            'layout': {
                'title': 'Message Cloud',
                'width': '600',
                'height': '600',
                'automargin': 'True',
                'titlefont': {'size': '30'},
                'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Return message classification results
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001, debug=True)

if __name__ == '__main__':
    main()
