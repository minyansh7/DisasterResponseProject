# Disaster Response Project

## Table of Contents

- [Project Overview](#Project-Overview)
- [Project Components](#Project-Components)
- [Project Installation](#Project-Installation)
- [Project Deployment](#Project-Deployment)
- [Live Demo](#Live-Demo)

## Project Overview

In this project, it analyze thousands of real messages that were sent through natural disasters, either through social media or directly to disaster response organizations.

A ETL pipeline is built to precess message and category data from csv files, and load them into a SQLite database.

A machine learning pipeline is created to categorize events that sent classified messages to appropriate disaster relief agencies.

The project also includes the web app  that provides visualizations, and help emergency workers to classify new messages for 36 categories. 

Machine learning are critical to help different organizations to understand which messages are relevant to them, and which messages to prioritize.

## Project Components
There are three components for this project.

1. ETL Pipeline
`process_data.py`, a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
`train_classifier.py`, a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
the flask web app:
- Add data visualizations using Plotly in the web app. 

## Project Installation
#### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

## Project Deployment
1. To run the web app, go into the Terminal and type:
`cd project folder path`
`python app/run.py`

Make sure that the web app is working locally.

(Run the web app inside project folder. In the terminal, use this command to get the link for vieweing the app:
`env | grep WORK`

The link wil be:
http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.
)

2. Next, go to www.heroku.com and create an account if you haven't already.

3. In the terminal, update python using the terminal command `conda update python`

4. Create a virtual environment. Note that you can create the virtual environment inside the project folder. But then you would end up uploading that folder to Heroku unecessarily. Consider creating the virtual environment in the workspace folder. Or alternatively, you can create a .gitignore file inside the project folder so that the virtual enviornment folder gets ignored

5. pip install the libraries needed for the web app. In this case those are flask, pandas, plotly, and gunicorn.

6. next install the heroku command line tools with the following command:
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
https://devcenter.heroku.com/articles/heroku-cli#standalone-installation

7. then check the installation with the command:
`heroku —-version`

8. next, log into heroku using the command:
`heroku login`
and then enter your email and password when asked

9. remove app.run() from the run.py file

10. go into the deployment folder with:
`cd project folder path`

11. create a procfile with the command
`touch Procfile`
and put the following in the Procfile
`web gunicorn run:DisasterResponseApp`

To deploy the ML model we need to create 2 files. The first one is Procfile (no file extension) in this we will write “web: gunicorn run:DisasterResponseApp”. The web depicts that this is a web app and gunicorn is the server on which our app will run. The follwoing "run" represents the file name from where the Web App should start. The second part "DisasterResponseApp" represents the name of the app.


12. Then create a requirements file with this command:
`pip freeze > requirements.txt`

13. Next, initialize a git repository with the following commands:
`git init`
`git add .`

14. configure the email and user name, you can use these commands:
`git config --global user.email email@example.com`
`git config --global user.name "my name"`

15. make a commit with this command:
`git commit -m "first commit"`

16. create a uniquely named heroku app. Use this command:
`heroku create my-app-name`
If you get a message that the app name is already taken, try again with a different app name until you find one that is not taken

17. check that heroku added a remote repository with this command:
`git remote -v`

18. push the app to Heroku:
`git push heroku master`

Go to the link for your web app to see if it's working. The link should be https://app-name.heroku.com

## Live Demo

