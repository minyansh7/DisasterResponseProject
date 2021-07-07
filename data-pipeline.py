# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(data_file1,data_file2):
    # read in file
    
    messages = pd.read_csv(data_file1)
    categories = pd.read_csv(data_file2)
    df = messages.merge(categories,how='outer',on=['id'])

    # clean data
    
    #create a dataframw of the 36 individual category columns
    categories=df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    categories.columns=list(map(lambda x: x[ : -2], row))
    #Iterate through the category columns in df 
    #keep only the last character of each string (the 1 or 0)
    for column in categories:
        categories[column] = categories[column].map(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column].astype(str))
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    # concatenate df and categories dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)

    # load to database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse.db', engine, index=False)

    # define features and label arrays


    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
