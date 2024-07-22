# This Python 3 environment comes with many 
# helpful analytics libraries installed
# It is defined by the kaggle/python Docker image:
# https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will 
# list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that 
# gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be 
# saved outside of the current session

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Read the train data
X = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/train.csv',
    index_col='Id')

# Read test data
X_test_full = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/test.csv', 
    index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively 
# low cardinality (convenient but arbitrary)
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

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


def make_mi_scores(X, y):
    '''This function is used for calculating mutual information scores'''
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(
        X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    '''This function is used to visualise mutual information scores'''
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
	
# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    '''This function is used to calculate the mean absolute error of a model'''
    model = XGBRegressor(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Get names of columns with missing values in X_train
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
					 
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Extension to imputation
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(
    imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

mi_scores = make_mi_scores(imputed_X_train_plus, y_train)

print(mi_scores.head(20))
#print(mi_scores.tail(20))  # uncomment to see bottom 20

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
# plot_mi_scores(mi_scores.tail(20))  # uncomment to see bottom 20

# BASELINE MODEL!!!!!
# Define the baseline model with default parameters
my_model_1 = XGBRegressor(random_state=0) 

# Fit the model
my_model_1.fit(imputed_X_train_plus, y_train)

# Get predictions
predictions_1 = my_model_1.predict(imputed_X_valid_plus) 

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid) 

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)

def best_XGB_param(x):
    '''This function is used for tuning the parameters of the 
    XGBRegressor model. At least one parameter must be assigned 
    to a parameter in XGBRegressor. For example, learning_rate = x'''
    my_modelo = XGBRegressor(n_estimators=800, learning_rate=x, 
        random_state=0, max_depth=5, min_child_weight=5, gamma=0, 
        subsample = 0.7, colsample_bytree=0.9, scale_pos_weight=1, 
        reg_alpha=1e-5) 

    # Fit the model
    my_modelo.fit(imputed_X_train_plus, y_train) 

    # Get predictions
    predictions_2 = my_modelo.predict(imputed_X_valid_plus)
    # Calculate MAE
    mae_2 = mean_absolute_error(predictions_2, y_valid) 

    print("Mean Absolute Error:" , mae_2, "learning_rate:", x)
	
# MAIN MODEL!!!
# Define the main model
my_model_2 = XGBRegressor(n_estimators=800, learning_rate=0.05, 
    random_state=0, max_depth=5, min_child_weight=2, gamma=0, 
    subsample = 0.8, colsample_bytree=0.75, scale_pos_weight=1, 
    reg_alpha=0.01) 

# Fit the model
my_model_2.fit(imputed_X_train_plus, y_train) 

# Get predictions
predictions_2 = my_model_2.predict(imputed_X_valid_plus)
# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid) 

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)

zenith = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/train.csv', 
    index_col='Id')

x2 = zenith.copy()
y2 = x2.pop('SalePrice')

# One-hot encode the entire training data (to shorten the code, we use pandas)
x2 = pd.get_dummies(x2)
print("Shape of Train dataset:",x2.shape)

# One-hot encode the entire test data 
X_test_full = pd.get_dummies(X_test_full)
print("Shape of Test dataset:",X_test_full.shape)

# Get names of columns with missing values
cols_with_missing3 = [col for col in x2.columns
                     if x2[col].isnull().any()]

#extension to imputation on the whole training dataset
# Make copy to avoid changing original data (when imputing)
x2_plus = x2.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing3:
    x2_plus[col + '_was_missing'] = x2_plus[col].isnull()

# Imputation for train dataset
my_imputer = SimpleImputer()
imputed_x2_plus = pd.DataFrame(my_imputer.fit_transform(x2_plus))

# Imputation for train dataset removed column names; put them back
imputed_x2_plus.columns = x2_plus.columns

# Get names of columns with missing values
cols_with_missing2 = [col for col in X_test_full.columns
                     if X_test_full[col].isnull().any()]

# Make copy to avoid changing original data (when imputing)
X_test_full_plus = X_test_full.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing2:
    X_test_full_plus[col + '_was_missing'] = X_test_full_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_test_full_plus = pd.DataFrame(
    my_imputer.fit_transform(X_test_full_plus))

# Imputation removed column names; put them back
imputed_X_test_full_plus.columns = X_test_full_plus.columns

# ensures missing columns in test and present in train data are added 
x2_cols = list(imputed_x2_plus.columns)
X_test_full_cols = list(imputed_X_test_full_plus.columns)
cols_not_in_test = {c:0 for c in x2_cols if c not in X_test_full_cols}
imputed_X_test_full_plus = imputed_X_test_full_plus.assign(**cols_not_in_test)


# ensures missing columns in train and present in test data are added 
cols_not_in_train = {c:0 for c in X_test_full_cols if c not in x2_cols}
imputed_x2_plus = imputed_x2_plus.assign(**cols_not_in_train)

# Columns are no longer in the same order after adding 
# missing columns to train and test data therefore columns 
# in each dataset need to be reordered so that they are identical
imputed_x2_plus = imputed_x2_plus.reindex(
    sorted(imputed_x2_plus.columns), axis = 1)
imputed_X_test_full_plus = imputed_X_test_full_plus.reindex(
    sorted(imputed_X_test_full_plus.columns), axis = 1)
imputed_X_test_full_plus.head()

# There was an error due to a feature shape mismatch since there 
# are an unequal number of categories being OH-Encoded
# Solution: find the number of missing columns in both imputed datasets
# and append the missing columns to each dataset respectively

# Define the final model
final_model = XGBRegressor(n_estimators=800, learning_rate=0.05, 
    random_state=0, max_depth=5, min_child_weight=2, gamma=0, 
    subsample = 0.8, colsample_bytree=0.75, scale_pos_weight=1, reg_alpha=0.01)

# Fit the final model
final_model.fit(imputed_x2_plus, y2) 

# Get predictions
predictions_final = final_model.predict(imputed_X_test_full_plus)

# need to reset the index to the original index of the test data to be able 
# to submit to competition
imputed_X_test_full_plus.index =X_test_full.index
imputed_X_test_full_plus.head()

# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': imputed_X_test_full_plus.index,
                       'SalePrice': predictions_final})
output.to_csv('submission.csv', index=False)