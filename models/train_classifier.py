import sys
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# category_names = ['related', 'request', 'offer',
#        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
#        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
#        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
#        'infrastructure_related', 'transport', 'buildings', 'electricity',
#        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
#        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
#        'other_weather', 'direct_report']



# def load_data(database_filepath):
#     """
#     Loads data from database

#     Args:
#         database_filepath: path to database

#     Returns:
#         (DataFrame) X: feature
#         (DataFrame) Y: labels

#     """
#     engine = create_engine('sqlite:///' + database_filepath)
#     df = pd.read_sql_query('select * from cleanData', engine)

#     X = df['message'].values
#     y = df.drop(['id','message','original','genre'], axis=1)
#     category_names = y.columns
#     return X, y, category_names

def load_data(database_filepath):
    # table name
    table_name = 'disaster'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    # X = df['message']
    # y = df[category_names].values
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

# def tokenize(text):
#     """
#     Tokenizes a given text.

#     Args:
#         text: text string

#     Returns:
#         (str[]): array of clean tokens

#     """
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

def tokenize(text, lemma=True, remove_stop=True, remove_short=True):
    '''
    input: (
        text: 
        lemma: True if lemmatize word, 
        remove_stop: True if remove stopwords  
        remove_short: True if remove short words > 2
            )
    Function tokenizes the text 
    output: (
        returns cleaned tokens in list 
            )
    '''
    # stopword list 
    STOPWORDS = list(set(stopwords.words('english')))
    # initialize lemmatier
    lemmatizer = WordNetLemmatizer()

    # split string into words (tokens)
    tokens = word_tokenize(text)
    # remove short words
    if remove_short: tokens = [token for token in tokens if len(token) > 2]
    # put words into base form
    if lemma: tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    # remove stopwords
    if remove_stop: tokens = [token for token in tokens if token not in STOPWORDS]
    # return data 
    return tokens


def build_model():
    """Builds classification model """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    param_grid = {
        # 'vect__ngram_range': ((1, 1)),
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=1, cv=None)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset

    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle

    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

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
