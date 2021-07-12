import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import re
import nltk
import seaborn as sns
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
#TfidfVectorizer = CountVectorizer + TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_data(database_filepath):
    '''
    INPUT:
        database_filepath:relative filepath of database
    OUTPUT:
        X: the original messages
        Y: the classified results of disaster response
        category_name:the list of names for Y columns
    DESCRIPTION:
        The function is to load data from database, and assign corrected collumns to X,Y
    
    '''
    engine = create_engine('database_filepath')
    df=pd.read_sql_table('Message',con=engine)
    X=df['message']
    Y = df.iloc[ : , -36:]
    Y = Y.drop(['related','child_alone'],axis=1)
    category_name = Y.columns
    return X,Y,category_name

def tokenize(text):
    '''
    INPUT:
        text:raw message
    OUTPUT:
        clean_tokens:tokenized words
    DESCRIPTION:
        The function is to process the scentence, normalize texts, tokenize texts.
        Convert all cases to lower cases, remove extra space,stop words, and 
        reduce words to their root form.
    '''
    clean_tokens=[]    
    #remove punctuation,normalize case to lower cases, and remove extra space
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower()).strip()
    
    #tokenize text
    tokens=word_tokenize(text)
    
    for w in tokens:  
        #remove stop words
        if w not in stopwords.words("english"):
        #lemmatization
        #reduce words to their root form
            lemmed = WordNetLemmatizer().lemmatize(w)
            clean_tokens.append(lemmed)
    return clean_tokens


def build_model():
    '''
    Build a machine learning model using the pipeline
    OUTPUT:
        CV: the model with the optimized parameters
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(n_jobs = -1,random_state=6)))])
    parameters = {
        #'vectorizer__ngram_range':((1,1),(1,2)),#the range for a string of n words
        #'tfidy_smooth_idf':[True,False],
        'clf__estimator__l1_ratio': [0.1,0.15,0.2,0.25,0.3,0.35],
        'clf__estimator__alpha': [0.0001,0.001,0.01,0.1],
        'clf__estimator__penalty':['l2','l1','elasticnet','none'],
        'clf__estimator__loss':['hinge','log','squared hinge','modified_huber','perceptron']
    }
    cv = GridSearchCV(pipeline,parameters,cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_name):
    '''
    print out classification report and accuracy scode for the best model result
    '''
    Y_test_pred = model.predict(X_test)
    print('Accuracy score:\n'.format(accuracy_score(Y_test, Y_test_pred)))
    try:
        print(classification_report(Y_test, Y_pred, target_names=category_name))
    except:
        print('classification report on whole dataset is not working.')

def save_model(model, model_filepath):
    '''
    Export the model as a pickle file
    '''
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_name = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=6)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_name)

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