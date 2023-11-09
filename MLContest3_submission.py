# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:35:27 2023

@author: rghosh
"""

#pip install catboost

# Import pandas as pd
import pandas as pd 
import numpy as np
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold

## Updating the Directory and just running the Entire Script should work 
os.chdir("C:/Users/rghosh/Documents/Graduate Curicullum/Spring'23/STAT 8456/Contest 3/contest3data")



train=pd.read_csv("train.csv")

###########################
# Convert the orderDate and deliveryDate columns to datetime objects
train['orderDate'] = pd.to_datetime(train['orderDate'])
train['orderMonth'] = train['orderDate'].dt.month
train['orderWeek'] = train['orderDate'].dt.week
train['orderDayOfWeek'] = train['orderDate'].dt.dayofweek

train.loc[train['orderWeek'] == 52, 'orderWeek'] = 1

# Sort values by 'orderDate'
train['orderDate'] = pd.to_datetime(train['orderDate'])
train = train.sort_values(by='orderDate')

# 1. Number of items in the order
train['items_in_order'] = train.groupby('orderID')['itemID'].transform('count')

# 2. Check for duplicates in order
train['duplicates_in_order'] = train.duplicated(subset=['orderID', 'itemID'], keep=False)

# 3. Check if the customer has made a return before
train['previous_returns'] = train.sort_values(by='orderDate').groupby('customerID')['return'].apply(lambda x: x.shift().cumsum().fillna(0))

# 4. Check if a voucher was used
train['used_voucher'] = train['voucherID'] != 'NONE'

# 5. Check if there were previous returns made with the voucher used
# Create a temporary column 'voucher_return' which is 1 if both 'return' and 'used_voucher' are 1, else 0
train['voucher_return'] = (train['return'] & train['used_voucher']).astype(int)
train['previous_voucher_returns'] = train.sort_values(by='orderDate').groupby('voucherID')['voucher_return'].apply(lambda x: x.shift().cumsum().fillna(0))

# Drop temporary 'voucher_return' column
train = train.drop(columns='voucher_return')

train = train.sort_values(by='orderDate')

# Create a list of object columns
object_cols = ['itemID','orderID', 'colorCode', 'sizeCode','customerID','typeCode','voucherID','deviceCode', 'paymentCode']
#object_cols = ['itemID', 'size', 'color','manufacturerID','customerID','salutation','state','return']
#'return' to be in numeric 

# Use apply() to convert object columns to categorical data type
train[object_cols] = train[object_cols].apply(lambda x: x.astype('category'))


X = train[['itemID','sizeCode', 'voucherID','deviceCode', 'paymentCode','customerID',
           'colorCode','typeCode','price','recommendedPrice','voucherAmount','used_voucher','previous_voucher_returns',
           'previous_returns','duplicates_in_order','items_in_order']]
y = train['return']

numerical_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()

# Cast categoricals to string - CatBoost Requirement
X[categorical_features] = X[categorical_features].astype(str)


############################  Prepare the Test Set ################
df_train=pd.read_csv("train.csv")
test= pd.read_csv("test.csv")

###########################

# Reset indices
#df_train = df_train.reset_index(drop=True)
# test = test.reset_index(drop=True)

# Set 'recordID' as index in both dataframes
df_train = df_train.set_index('recordID')
test = test.set_index('recordID')


# Add placeholder 'return' column to the test set
# test['return'] = np.nan

# Add placeholder 'return' column to the test set
test['return'] = False

# Concatenate test and test sets
df_combined = pd.concat([df_train, test])

df_combined['orderDate'] = pd.to_datetime(df_combined['orderDate'])

# Sort by 'orderDate'
df_combined = df_combined.sort_values(by='orderDate')

# Compute 'previous_returns' feature
df_combined['previous_returns'] = df_combined.groupby('customerID')['return'].apply(lambda x: x.shift().cumsum().fillna(0))

# Check if a voucher was used
df_combined['used_voucher'] = df_combined['voucherID'] != 'NONE'

# Create a temporary column 'voucher_return' which is 1 if both 'return' and 'used_voucher' are 1, else 0
df_combined['voucher_return'] = (df_combined['return'] & df_combined['used_voucher']).astype(int)

# Compute 'previous_voucher_returns' feature
df_combined['previous_voucher_returns'] = df_combined.groupby('voucherID')['voucher_return'].apply(lambda x: x.shift().cumsum().fillna(0))

# Split back into test set
# test = df_combined[df_combined['return'].isna()].reset_index(drop=True)

# test= df_combined[df_combined['return'] == False].reset_index(drop=True)

#test = df_combined[df_combined.index.isin(test.index)]

test = df_combined.loc[test.index]

# Drop temporary 'voucher_return' and placeholder 'return' in test set

test = test.drop(columns=['voucher_return', 'return'])


# Convert the orderDate and deliveryDate columns to datetime objects
test['orderDate'] = pd.to_datetime(test['orderDate'])
test['orderMonth'] = test['orderDate'].dt.month
test['orderWeek'] = test['orderDate'].dt.week
test['orderDayOfWeek'] = test['orderDate'].dt.dayofweek

# Sort values by 'orderDate'
test['orderDate'] = pd.to_datetime(test['orderDate'])
test = test.sort_values(by='orderDate')

# Number of items in the order
test['items_in_order'] = test.groupby('orderID')['itemID'].transform('count')

# Check for duplicates in order
test['duplicates_in_order'] = test.duplicated(subset=['orderID', 'itemID'], keep=False)





# Create a list of object columns
object_cols = ['itemID','orderID', 'colorCode', 'sizeCode','customerID','typeCode','voucherID','deviceCode', 'paymentCode']
#object_cols = ['itemID', 'size', 'color','manufacturerID','customerID','salutation','state','return']
#'return' to be in numeric 

# Use apply() to convert object columns to categorical data type
test[object_cols] = test[object_cols].apply(lambda x: x.astype('category'))

## more work needed on cleaning color, size and salutation string to float error 

X_final_test = test[['itemID','sizeCode', 'voucherID','deviceCode', 'paymentCode','customerID',
           'colorCode','typeCode','price','recommendedPrice','voucherAmount','used_voucher','previous_voucher_returns',
           'previous_returns','duplicates_in_order','items_in_order']]


# Cast categoricals to string
X_final_test[categorical_features] = X_final_test[categorical_features].astype(str)

from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# Define the categorical features for the CatBoost model
cat_features_index = [i for i, col in enumerate(X.columns) if col in categorical_features]

# Initialize CatBoostClassifier
model = CatBoostClassifier(od_type='Iter',
                           od_wait=100,
                           loss_function="Logloss",
                           eval_metric='Accuracy',
                           verbose=True,task_type='CPU')
param_dist = {'iterations': sp_randInt(1000, 2000),
              'depth': sp_randInt(3, 10),
              'learning_rate': sp_randFloat(0.001, 0.1),
              'l2_leaf_reg': sp_randInt(1, 10),
              'border_count': sp_randInt(32, 256)}


# Instantiate the grid search object
random_search = RandomizedSearchCV(estimator=model, 
                                   param_distributions=param_dist,
                                   n_iter= 25, # number of parameter settings sampled
                                   cv=3, # number of cross-validation folds
                                   scoring='accuracy',
                                   n_jobs=-1)


# Define the groups based on customerID
groups = X['customerID']

# Initialize GroupKFold
gkf = GroupKFold(n_splits=5)

# List to store best models from each iteration
best_models = []

# Initialize dataframes to store predictions and probabilities
predictions_df = pd.DataFrame()
probabilities_df = pd.DataFrame()

# Iterate over each split
for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the grid search object to the data
    random_search.fit(X_train, y_train, cat_features=cat_features_index, eval_set=(X_test, y_test))

    # Get the best parameters and score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best parameters found: ", best_params)
    print("Best accuracy found: ", best_score)

    # Get the best estimator and predict
    best_model = random_search.best_estimator_
    best_models.append(best_model)

    preds_class = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_class)
    print("CatBoost Model Accuracy (Test Set): {:.2f}%".format(accuracy*100))

    # Make predictions on the X_final_test set
    preds_final_test = best_model.predict(X_final_test)
    preds_proba_final_test = best_model.predict_proba(X_final_test)[:, 1]

    # Store the predictions and probabilities in the dataframes
    predictions_df['model_{}'.format(i)] = preds_final_test
    probabilities_df['model_{}'.format(i)] = preds_proba_final_test


predictions_df.to_csv('cat_dt_pred2.csv')
probabilities_df.to_csv('cat_dt_prob2.csv')

#create submission
submission = pd.DataFrame()
submission['recordID'] = test['recordID']
submission['return'] = predictions_df.apply(lambda row: 1 if sum(row) > 2 else 0, axis=1)
#submission['return'] = probabilities_df.apply(lambda row: 1 if np.mean(row) > 0.5 else 0, axis=1)


submission.to_csv('submission.csv', index=False)