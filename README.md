# Titanic dataset 
## Description
My solution to Kaggle's Titanic dataset (binary classification task), using Random Forest Classifier.

Cross-validation accuracy scored 82%

Public score (Test accuracy): 79.4%

Ranking: ~2500

## Files
Main file: Titanic_kaggle.py 

Submitted file (2nd submission): Submission_2_82percent.csv 

## Methods
Some basic EDA and feature selection is performed. Categorical variables are transformed to a one-hot representation. Classification performed by sklearn.RandomForestClassifier. Hyperparameters are tuned by sklearn.model_selection.GridSearchCV.  
