import sys
# import libraries
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle


def load_data(database_filepath):
    """
    load in data from sql lite database,
    into X,Y and category name values
    """


    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("dataframe",engine)
    X = df["message"]
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', \
        'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', \
            'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',\
                 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings',\
                      'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',\
                           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    Y = df[category_names]

    return X,Y,category_names

def tokenize(text):
    """
    tokenize words, lemmatize,and remove stop words
    """
    #tokenize the sentence
    tokens = word_tokenize(text)
    
    #normalize and lemmatize words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return cleaned_tokens


def build_model():
    """
    create the pipeline for the model
    """
    pipeline = Pipeline([
    ("vect",CountVectorizer(tokenizer = tokenize)),
    ("tfidf",TfidfTransformer()),
    #use all available threads
    ("pred",MultiOutputClassifier(XGBClassifier(nthread = -1)))
    ])

    #cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    print out l1, recall, and precision
    metrics for the model to evaluate performance
    """
    #make prediciton
    y_pred = pd.DataFrame(model.predict(X_test), columns = Y_test.columns)

    #generate report for each column
    #total f1-score
    total = 0
    for column in Y_test.columns.values:
        
        report = classification_report(Y_test[column], y_pred[column])
        f1_score = report.split()[-2]
        total += float(f1_score)
        print(column)
        print(report)

    #print out total average accuracy
    average_accuracy = total / len(Y_test.columns)
    print("The total average accuracy is: ",average_accuracy)


def save_model(model, model_filepath):
    """
    pickle the model for 
    later use
    """
    #pickle model
    with open(model_filepath,"wb") as pickle_file:
        pickle.dump(model,pickle_file)

def main():
    """
    main function running
    all necessary functions
    """
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