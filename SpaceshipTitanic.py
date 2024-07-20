# This Python 3 environment comes with many 
# helpful analytics libraries installed.
# It is defined by the kaggle/python Docker 
# image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that 
# gets preserved as output when you create a version using "Save & Run All".
# You can also write temporary files to /kaggle/temp/, but they won't be saved
# outside of the current session

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Read the train data
X = pd.read_csv(
    '/kaggle/input/spaceship-titanic/train.csv', 
    index_col='PassengerId')

# Read test data
X_test_full = pd.read_csv(
    '/kaggle/input/spaceship-titanic/test.csv', 
    index_col='PassengerId')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Transported'], inplace=True)
y = X.Transported     
X.drop(['Transported'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality 
#(convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns 
    if X_train_full[cname].nunique() < 10 
    and X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns 
    if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# Preprocessing for numerical data
numeric_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, low_cardinality_cols)
    ])

model = RandomForestClassifier(n_estimators=211, max_depth=10)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

my_pipeline.score(X_train, y_train)

param_dist = {'model__n_estimators': randint(50,500),
              'model__max_depth': randint(1,20)}


# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(my_pipeline, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Define the final random forest model

best_rf.fit(X,y)
predictions_final = best_rf.predict(X_test)

best_rf.score(X, y)

output = pd.DataFrame({'PassengerId': X_test_full.index,
                       'Transported': predictions_final})
output.to_csv('submission.csv', index=False)