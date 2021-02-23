import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load in csv data into a merged DataFrame,
    and prepare to be cleaned and loaded into save data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = categories.merge(right = messages, on = "id")
    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(pat = ";",expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_col_vals = row.values
    category_colnames = []
    for title in category_col_vals:
        category_colnames.append(title[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    #use lambda function
    categories = categories.apply(lambda x : (x.str[-1]).astype("float"))

    # drop the original categories column from `df`
    df.drop(columns = ["categories"],inplace = True)    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)
    return df



def clean_data(df):
    """
    clean data by removing duplicates, and dropping any rows containing none values
    """
    df = df.drop_duplicates()
    
    df = df.dropna()
    #assume 2 means 1, replace all 2's with 1's
    df = df.replace(to_replace = 2, value = 1)
    return df


def save_data(df, database_filename):
    """
    save the data to an sqllite db
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('dataframe', engine, index=False,if_exists='replace')  


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