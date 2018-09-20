import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
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
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    target_distribution = df.drop(['id','message','original','genre'], axis=1).mean()
    target_names = list(target_distribution.index)

    counter = Counter()
    for message in df.message.values:
        tokens = tokenize(message)
        for token in tokens:
            counter[token] += 1
    top = counter.most_common(20)
    top_words = [word[0] for word in top]
    top_counts = [count[1] for count in top]

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
                    x=[ 1.32387431,  2.08095859,  2.28200191,  3.84280063,  1.06846261,
        1.75591011,  1.09543713,  1.21952394,  1.06488479,  1.09092727,
        2.44971265,  0.85046083,  1.30588492,  0.89535768,  0.86019524,
        0.95806966,  0.92903218,  0.77095894,  4.96078166,  1.12196315,
        2.15203571,  3.48018191,  1.56440752, 14.71221206,  2.18316589,
        1.59984597,  2.29044498,  1.83897773,  2.79469509,  1.42192902,
        0.63912257, 13.0668691 ,  0.87138158,  1.25398232,  2.38806614,
        0.61709328,  0.96743355,  2.99107884,  0.84541937,  1.3670957 ,
        3.50124604,  0.7603816 ,  4.31921443,  1.36427637,  0.89703351,
        1.0846022 ,  1.62627013,  0.73282301,  0.93400346,  1.52460199,
        1.07487004,  0.68203542,  0.62955113, 13.20132975,  1.00618001,
        0.88927837,  1.87984696,  0.77801555,  9.9382635 ,  9.21383468,
        0.76249731,  1.22801256,  0.9969835 ,  1.59275064,  1.07145118,
        1.52747895,  0.86957524,  1.07124004,  1.16548356,  0.58448419,
        0.77112348,  1.4024441 ,  1.63217883,  0.80136827,  1.84756325,
        1.27928537,  0.89859216,  1.03514028,  0.7028156 ,  1.63522788,
        2.22187879,  0.86319615,  0.87086264,  0.96812632,  0.78597381,
        3.63059706,  2.27572388,  1.28694163,  0.61667733,  0.97017542,
        1.10230073,  1.66794016,  1.24812908,  1.48237197, 12.11203306,
        0.68591323,  1.40400648,  1.38583517,  4.14875403,  1.44380408],
                    y=[ 1.36173436e-01,  1.43321458e-01, -4.49539155e-01, -3.32974239e-01,
       -9.97572352e-02, -2.37572092e-01,  1.62959035e-01,  3.51663322e-01,
       -1.02709365e-01,  2.88544196e-01,  1.21745584e+00, -3.22986046e-01,
        3.53646932e-01,  1.52445606e-01,  3.02526138e-01, -3.97306348e-01,
       -1.99530598e-01, -1.04413587e-01,  1.68425110e+01, -3.05015818e-02,
        3.47526900e-02,  8.91854620e-01, -1.62948405e-01, -1.02685702e+01,
        7.16759271e-01,  7.08192804e-01,  9.34727083e-01,  3.61159929e-01,
        9.69627898e+00, -3.61308988e-01,  3.96664472e-02,  3.49896777e+00,
        1.26021688e+00,  8.57366383e-02,  6.26048650e-01,  2.21492704e-01,
       -1.75551210e-01,  4.40752157e+00,  1.64137946e-01,  6.56240643e-01,
        4.20985979e+00,  4.21666628e-01,  6.61546924e+00,  2.19842379e-01,
       -7.60720935e-02,  4.47275184e-01,  4.93675702e-01,  2.97426329e-01,
       -2.86327815e-01,  8.51906703e-01,  9.84189147e-02,  1.35612648e-01,
        1.56836170e-01, -2.89321208e+00,  5.35801514e-01,  2.96865726e-01,
        8.72556306e-01,  1.02663152e-01,  2.27117617e+00,  3.14462147e+00,
       -4.61469622e-02, -3.83405515e-02,  1.79733122e-01,  6.61174021e-02,
        2.29062276e-01,  3.10022157e-01, -4.14372072e-01, -1.18947741e-01,
        2.03506952e-02,  7.04704875e-01,  4.18756038e-01,  6.52005985e-01,
        5.27730883e-01,  2.21248522e-01, -5.05884729e-01,  4.19978575e-01,
        2.96575668e-01,  2.34976685e-01,  1.32482314e-01, -5.30469290e-02,
       -1.40713605e+00, -1.16613113e-02, -3.80556449e-01,  4.72392034e-01,
       -3.33576678e-02, -1.23025079e+00,  9.33441398e-01,  5.00111598e-01,
        7.94735400e-02,  5.74043409e-01,  1.73702760e-01,  8.59824850e-01,
       -1.45344692e-01,  1.21412722e+00, -1.01013502e+01,  2.29488928e-01,
       -3.43861632e-01,  3.49207358e-01,  6.00377053e+00,  5.94592859e-01],
                    mode='markers+text',
                    name='Markers and Text',
                    text=['affected', 'aid', 'also', 'area', 'assistance', 'child', 'city', 'come', 'community', 'could', 'country', 'crop', 'day', 'destroyed', 'disaster', 'disease', 'district', 'due', 'earthquake', 'emergency', 'family', 'find', 'flood', 'food', 'get', 'give', 'good', 'government', 'haiti', 'health', 'heavy', 'help', 'hit', 'home', 'house', 'hurricane', 'including', 'information', 'international', 'job', 'know', 'last', 'like', 'live', 'local', 'make', 'many', 'may', 'medical', 'message', 'million', 'month', 'national', 'need', 'new', 'number', 'one', 'part', 'people', 'please', 'power', 'problem', 'province', 'rain', 'region', 'relief', 'river', 'road', 'sandy', 'santiago', 'say', 'school', 'send', 'service', 'shelter', 'since', 'state', 'still', 'storm', 'street', 'supply', 'support', 'system', 'take', 'team', 'tent', 'thank', 'thanks', 'three', 'time', 'two', 'victim', 'village', 'want', 'water', 'week', 'well', 'work', 'would', 'year'],
                    textposition='top center'
                )
            ],

            'layout': {
                'title': 'Latent Semantic Analysis (LSA)',
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