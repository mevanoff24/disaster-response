import sys
import pandas as pd
import numpy as np
import nltk 
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    input: (
        messages_filepath: csv file 
        categories_filepath: csv file 
            )
    reads in two csv files and outputs a merged dataframe
    output: (pandas dataframe)
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    # return df 
    return df


def clean_data(df):
    '''
    input: (
        df: pandas dataframe 
            )
    reads in a pandas dataframe and formats data for ML model 
    output: (pandas dataframe)
    '''
    # # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # plucks out first row
    row = categories[:1]
    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    assert len(df[df.duplicated()]) == 0
    # return df
    return df


def save_data(df, database_filename):
    '''
    input: (
        df: pandas dataframe 
        database_filename: database filename
            )
    Saves pandas dataframe to database
    output: (None)
    '''
    # table name -- ideally this should be another input param
    table_name = 'disaster'
    # create engine 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists 
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()