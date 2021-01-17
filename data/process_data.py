import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    # Messages file has 68 duplicates row
    messages.drop_duplicates(inplace=True)
    
    categories = pd.read_csv(categories_filepath)
    # categories file has 32 duplicates row
    categories.drop_duplicates(inplace=True)
    
    # In summary messages have 26180 unique rows and categories have 26216 uniq rows
    # Due to the inconsistence of inputs, merge will be conducted after splitting categories
    
    # create a dataframe of the 36 individual category columns
    categories_split = categories.categories.str.split(pat=";",n=-1, expand=True)
    
    # Use first row to get the column names
    row = categories_split.iloc[0]
    category_colnames = [name.split("-")[0] for name in row]
    categories_split.columns = category_colnames

    # Merge splitted categoreis with original categries dataframe to get id back
    new_categories = pd.concat([categories, categories_split], axis=1)
    new_categories.drop(labels='categories', axis=1, inplace=True)
    
    # set each value to be the last character of the string
    for column in category_colnames:
        new_categories[column] = new_categories[column].apply(lambda x:x.split('-')[-1]).astype(int)
    
    # merge datasets
    df = messages.merge(new_categories, on='id').reset_index(drop=True)
    
    # child_alone only have 1 value and it is droped here
    df.drop(labels=['child_alone'], axis=1, inplace=True)
    return df
    
def clean_data(df):
    # Duplicates removeal was conducted in load_data part
    # this function does not have any real function
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('rawdata', engine, if_exists='replace',index=False)


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