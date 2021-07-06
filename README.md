# DisasterResponseProject

### Project Overview

In this project, it analyze thousands of real messages that were sent through natural disasters, either through social media or directly to disaster response organizations.

A ETL pipeline is built to precess message and category data from csv files, and load them into a SQLite database.

A machine learning pipeline is created to categorize events that sent classified messages to appropriate disaster relief agencies.

The project also includes the web app  that provides visualizations, and help emergency workers to classify new messages for 36 categories. 

Machine learning are critical to help different organizations to understand which messages are relevant to them, and which messages to prioritize.

### Project Components
There are three components for this project.

1. ETL Pipeline
process_data.py, a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
train_classifier.py, a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
the flask web app:
- Add data visualizations using Plotly in the web app. 
