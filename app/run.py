import json
import plotly
import pandas as pd
import numpy as np 

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)


STOPWORDS = set(stopwords.words('english'))
STOPWORDS = STOPWORDS.union({'wa', 'said', 'ha', "n't", 'http'})


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if (len(tok) > 2) and (tok not in STOPWORDS):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def compute_LSA(messages, load=True, filepath='../data/lsa.npz'):
    if load:
        print('loading')
        data = np.load(filepath)
        return list(data['Z0']), list(data['Z1']), list(data['names'])
    else:
        vect = TfidfVectorizer(max_features=100, tokenizer=tokenize)
        X_tfidf = vect.fit_transform(messages)
        svd = TruncatedSVD()
        Z = svd.fit_transform(X_tfidf.toarray().T)
        np.savez(filepath, Z0=Z[:,0], Z1=Z[:,1], names=vect.get_feature_names())
        return Z[:,0], Z[:,1], vect.get_feature_names()


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # target distribution
    target_distribution = df.drop(['id','message','original','genre'], axis=1).mean()
    target_names = list(target_distribution.index)

    # top 20 words and counts 
    counter = Counter()
    for message in df.message.values:
        tokens = tokenize(message)
        for token in tokens:
            counter[token] += 1
    top = counter.most_common(20)
    top_words = [word[0] for word in top]
    top_counts = [count[1] for count in top]

    # lsa 
    Z0, Z1, names = compute_LSA(df.message.values)

    # create visuals
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
                }
            }
        },
        {
            'data': [
                Bar(
                    x=target_names,
                    y=target_distribution
                )
            ],

            'layout': {
                'title': 'Percentage of Needs',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Needs"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts
                )
            ],

            'layout': {
                'title': 'Top 20 words and counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=Z0,
                    y=Z1,
                    mode='markers+text',
                    name='Markers and Text',
                    text=names,
                    textposition='top center'
                )
            ],

            'layout': {
                'title': 'Word Similarities (LSA)',
                'yaxis': {
                    'title': "SVD1"
                },
                'xaxis': {
                    'title': "SVD2"
                }
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()