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

# stopwords
STOPWORDS = set(stopwords.words('english'))
# add some weird words, mess with visuals 
STOPWORDS = STOPWORDS.union({'wa', 'said', 'ha', "n't", 'http', 'also'})
# STOPWORDS = list(STOPWORDS)

def tokenize(text):
    '''
    input: (
        text: list or numpy array
            )
    Function lemmatizes and tokenizes the text based on stopwords and length of token
    output: (
        returns cleaned tokens in list 
            )
    '''
    # split string into words (tokens)    
    tokens = word_tokenize(text)
    # initialize lemmatier
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    # remove short words and stopwords
    for tok in tokens:
        if (len(tok) > 2) and (tok.lower() not in STOPWORDS):
            # put words into base form
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def compute_word_counts(messages, load=True, filepath='../data/counts.npz'):
    '''
    input: (
        messages: list or numpy array
        load: Boolean value if load or run model 
        filepath: filepath to save or load data
            )
    Function computes the top 20 words in the dataset with counts of each term
    output: (
        top_words: list
        top_counts: list 
            )
    '''
    if load:
        # load arrays
        data = np.load(filepath)
        return list(data['top_words']), list(data['top_counts'])
    else:
        # get top words 
        counter = Counter()
        for message in messages:
            tokens = tokenize(message)
            for token in tokens:
                counter[token] += 1
        # top 20 words 
        top = counter.most_common(20)
        top_words = [word[0] for word in top]
        top_counts = [count[1] for count in top]
        # save arrays
        np.savez(filepath, top_words=top_words, top_counts=top_counts)
        return list(top_words), list(top_counts)


def compute_LSA(messages, load=True, filepath='../data/lsa.npz'):
    '''
    input: (
        messages: list or numpy array
        load: Boolean value if load or run model 
        filepath: filepath to save or load data
            )
    Function computes Latent Semantic Analysis for creating a vector representation of text. 
    Gives a way to compare words for their similarity by calculating the distance between vectors.
    output: (
        list for first dimension of SVD
        list for second dimension of SVD
        names of features 
            )
    '''
    if load:
        # load arrays
        data = np.load(filepath)
        return list(data['Z0']), list(data['Z1']), list(data['names'])
    else:
        # Tf-Idf
        vect = TfidfVectorizer(max_features=100, tokenizer=tokenize)
        X_tfidf = vect.fit_transform(messages)
        # SVD 
        svd = TruncatedSVD()
        Z = svd.fit_transform(X_tfidf.toarray().T)
        # save arrays
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
    top_words, top_counts = compute_word_counts(df.message.values)

    # LSA
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
                'xaxis': dict(
                    autorange=True,
                    zeroline=False,
                    ticks='',
                    showticklabels=False
                ),
                'yaxis': dict(
                    autorange=True,
                    zeroline=False,
                    ticks='',
                    showticklabels=False
                )
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