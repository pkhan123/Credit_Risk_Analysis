# %% [markdown]
# # Credit Risk Analysis

# %% [markdown]
# ### For Google Colab

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# !cp drive/MyDrive/Credit_Risk_Analysis/* .

# %%
# !pip install catboost

# %% [markdown]
# ## Step-1: Import Libraries

# %% [markdown]
# ### Import the necessary base libraries

# %%
# Base libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os.path
import pickle

# %% [markdown]
# ### Import visualization libraries

# %%
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# %% [markdown]
# ### Import other libraries

# %%
# Othe libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# %% [markdown]
# ### Import helper functions

# %%
# import helper module
from Helper_Module_Credit_Risk_Analysis import *
Custom_Helper_Module()

# %% [markdown]
# ## Step-2: Import Dataset

# %%
dataset = pd.read_csv('Raw_Dataset/Loan_Dataset.csv')
# dataset = pd.read_csv('drive/MyDrive/Dataset/Credit_Risk_Analysis/Loan_Dataset.csv')

# %%
# copy the dataset into a new dataframe for further processing
imported_dataset = dataset.copy()

# %% [markdown]
# ## Step-3: Data Exploration

# %%
# display all columns
pd.options.display.max_columns = None

# %%
# shape of the dataset
imported_dataset.shape

# %%
# fast look of the data set
imported_dataset.head()

# %%
imported_dataset.describe(include='all')

# %%
Check_Missing_Values(imported_dataset)

# %%
# shape of the datset
imported_dataset.shape

# %%
categorical_variable_list = list(imported_dataset.select_dtypes(include=['object']).columns)
print(len(categorical_variable_list))
print(categorical_variable_list)

# %%
imported_dataset[categorical_variable_list].head()

# %%
# term can be converted to months
# emp_length can be converted to years
# issue_d, earliest_cr_line, last_pymnt_d, next_pymnt_d, last_credit_pull_d are in datetime format 
# and can be converted to months
# loan_status is our target variable and need some formatting

# %%
numerical_variable_list = list(imported_dataset.select_dtypes(include=['number']).columns)
print(len(numerical_variable_list))
print(numerical_variable_list)

# %%
Check_Missing_Values(imported_dataset)

# %%
# there are 18 features which have more than 80% missing values
# any technique to impute those missing values is most likely to indroduce errors
# we will simply drop them
high_missing_value_columns = ['mths_since_last_record', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']

# %%
imported_dataset.drop(high_missing_value_columns, axis = 1, inplace=True)

# %%
# certain features like id, member_id are not related to credit risk
# some features like recoveries, collection_recovery_fee are applicable only after default
# since our purpose is to predit the probability of default, we can drop those features
other_columns_to_drop = ['id', 'member_id', 'recoveries', 'collection_recovery_fee']

# %%
imported_dataset.drop(other_columns_to_drop, axis = 1, inplace=True)

# %%
imported_dataset.shape

# %% [markdown]
# ## Step-4: Format Dataset

# %% [markdown]
# ### Format the target variable

# %%
# since we want to predict probability of deafult, 'loan_status' is our target varible
# explore the unique values in 'loan_status' column
imported_dataset['loan_status'].value_counts(normalize = True)

# %%
# create a new column based on the loan_status column that will be our target variable
imported_dataset['good_loan'] = np.where(imported_dataset.loc[:, 'loan_status'].isin(['Charged Off', 
                                                                                            'Default', 
                                                                                            'Late (31-120 days)', 
                                                                                            'Does not meet the credit policy. Status:Charged Off']), 0, 1)

# %%
# drop the original 'loan_status' column
imported_dataset.drop(columns = ['loan_status'], inplace = True)

# %% [markdown]
# ### Convert datatime columns to months

# %%
# the folloing columns have datetime format
# 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'
# convert datetime columns to datetime format and 
# create a new column as a difference between today 
# and the respective date with prefix months_since_
# also drop the original column

Convert_Datetime_To_Months(imported_dataset, 'issue_d')
Convert_Datetime_To_Months(imported_dataset, 'earliest_cr_line')
Convert_Datetime_To_Months(imported_dataset, 'last_pymnt_d')
Convert_Datetime_To_Months(imported_dataset, 'next_pymnt_d')
Convert_Datetime_To_Months(imported_dataset, 'last_credit_pull_d')

# %% [markdown]
# ### Format some other columns

# %%
# convert loan tenure to months
Convert_Loan_Tenure_To_Months(imported_dataset, 'term')

# %%
# convert employment length to years
Convert_Employment_Length_To_Years(imported_dataset, 'emp_length')

# %%
formatted_dataset = imported_dataset.copy()

# %%
formatted_dataset.shape

# %% [markdown]
# ## Step-5: Train Test Split

# %%
# decoupling the dependent and independent variables
X = formatted_dataset.drop('good_loan', axis = 1)
y = formatted_dataset['good_loan']

# %%
# Train, Test split
# from now on till the model training we will only use X_train, y_train
# X_test and y_test will only be used during model testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# %%
X_train.head()

# %%
print(X_train.shape)
print(X_test.shape)

# %%
categorical_feature_list = list(X_train.select_dtypes(include=['object']).columns)
print(len(categorical_feature_list))
print(categorical_feature_list)

# %%
numerical_feature_list = list(X_train.select_dtypes(include=['number']).columns)
print(len(numerical_feature_list))
print(numerical_feature_list)

# %%
# backup copy the dataset for feature reference
backup_X_train = X_train.copy()
backup_X_test = X_test.copy()

# %% [markdown]
# ## Step-6: Handling Missing Values

# %%
Check_Missing_Values(X_train)

# %%
X_train.head()

# %%
Replace_Missing_Categorical_Values(X_train, categorical_feature_list)

# %%
Replace_Missing_Numerical_Values(X_train, numerical_feature_list)

# %%
X_train.head()

# %%
Check_Missing_Values(X_train)

# %%
X_train.shape

# %% [markdown]
# ## Step-5: Get Dummies

# %%
X_train.shape

# %%
X_train.describe(include='all')

# %%
# now there are some categorical features, where there are lot of categories
# we will take top 10 categories for each feature and replace the remaining categories by 'Other'

# %%
categorical_feature_list = list(X_train.select_dtypes(include=['object']).columns)
print(len(categorical_feature_list))
print(categorical_feature_list)

# %%
Reduce_Category(X_train, categorical_feature_list)

# %%
X_train.describe(include='all')

# %%
# now let's get the dummies
X_train = pd.get_dummies(X_train, drop_first=True)

# %%
X_train.shape

# %% [markdown]
# ## Step-8: Preliminary Feature Selection

# %%
# run time ~ 5 min
if not os.path.exists('preliminary_feature_list.txt'):
    Preliminary_Feature_Selection(X_train, y_train)

# %%
with open('preliminary_feature_list.txt') as f:
    preliminary_feature_list = f.read().splitlines()

# %%
preliminary_feature_list_X_train = X_train[preliminary_feature_list].copy()
X_train = preliminary_feature_list_X_train.copy()

# %%
X_train.columns

# %%
X_train.shape

# %%
X_train.head()

# %% [markdown]
# ## Step-9: Check Correlation and Multicollinearity

# %% [markdown]
# ### Get correlation qualified training dataset

# %%
X_train.shape

# %%
if not os.path.exists('correlation_qualified_feature_list.txt'):
    Check_Correlation(X_train)

# %%
with open('correlation_qualified_feature_list.txt') as f:
    correlation_qualified_feature_list = f.read().splitlines()

# %%
correlation_qualified_X_train = X_train[correlation_qualified_feature_list].copy()

# %%
X_train = correlation_qualified_X_train.copy()
X_train.shape

# %% [markdown]
# ### Get multicollinearity qualified training dataset

# %%
X_train.shape

# %%
numerical_feature_list = list(X_train.select_dtypes(include=['number']).columns)
print(len(numerical_feature_list))
print(numerical_feature_list)

# %%
if not os.path.exists('multicollinearity_qualified_feature_list.txt'):
    Check_Multicollinearity(X_train, numerical_feature_list)

# %%
with open('multicollinearity_qualified_feature_list.txt') as f:
    multicollinearity_qualified_feature_list = f.read().splitlines()

# %%
multicollinearity_qualified_X_train = X_train[multicollinearity_qualified_feature_list].copy()

# %%
X_train = multicollinearity_qualified_X_train.copy()
X_train.shape

# %%
X_train.head()

# %% [markdown]
# ## Step-9: Remove Outliers

# %%
numerical_feature_list = list(X_train.select_dtypes(include=['number']).columns)
print(len(numerical_feature_list))
print(numerical_feature_list)

# %%
Remove_Outlies(X_train, y_train, numerical_feature_list)

# %% [markdown]
# ## Step-10: Final Feature Selection

# %%
print(X_train.shape)
print(y_train.shape)

# %%
# now let's create feature selected dataset 
# Rrun time ~ 18 min
if not os.path.exists('selected_feature_list.txt'):
    # define parameters for feature selection
    max_validation_round = 5 # Range: 2-10, Default 10
    Make_Feature_Selection(X_train, y_train, max_validation_round)

# %%
with open('selected_feature_list.txt') as f:
    selected_feature_list = f.read().splitlines()

# %%
feature_selected_X_train = X_train[selected_feature_list].copy()

# %%
X_train = feature_selected_X_train.copy()
X_train.shape

# %% [markdown]
# ## Step-11: Build Model

# %%
# CatBoost Model
model_CBC = CatBoostClassifier(verbose=False)

# %%
# Randomized Search CV param_distributions for CatBoost

# iterations: int -> 100-1000 
iterations = [int(x) for x in np.linspace(100, 1000, num = 5)]
# learning_rate: float -> 0.01–0.30
learning_rate = [round(x, 2) for x in np.linspace(0.01, 0.30, num = 5)]
# depth: int -> 2–10 
depth = [int(x) for x in np.linspace(2, 10, num = 5)]
# l2_leaf_reg: int -> 2–30 
l2_leaf_reg = [int(x) for x in np.linspace(2, 30, num = 5)]
# border_count: int -> 10-100
border_count = [int(x) for x in np.linspace(10, 100, num = 5)]


# %%
# Create the random grid
random_param_distributions = {
                            'iterations': iterations,
                            'learning_rate': learning_rate,
                            'depth': depth,
                            'l2_leaf_reg': l2_leaf_reg,
                            'border_count': border_count
                            }
print(random_param_distributions)

# %%
# max_cross_validation
max_cross_validation = 2 # Test Value: 2, Default Value 10

# %%
# RandomizedSearchCV
randomized_search_cv = RandomizedSearchCV(
                                        estimator = model_CBC,
                                        param_distributions = random_param_distributions,
                                        n_iter = 20,
                                        cv = max_cross_validation, 
                                        random_state = 0,
                                        verbose = 1, 
                                        n_jobs = 1
                                        )

# %%
if not os.path.exists('catboost_classification_model.pkl'):
    # Run time: 17 min
    randomized_search_cv.fit(X_train, y_train)
    # Pickle the model
    with open('catboost_classification_model.pkl', 'wb') as file:
        pickle.dump(randomized_search_cv, file)

# %%
# load the pickle files
randomized_search_cv = pickle.load(open('catboost_classification_model.pkl','rb'))

# %%
model_best_score = randomized_search_cv.best_score_
print('Best score of the model: ', model_best_score)

# %%
randomized_search_cv.best_params_

# %% [markdown]
# ## Step-12: Model Evaluation

# %% [markdown]
# ### Model performance metrics for training dataset

# %%
y_hat = randomized_search_cv.predict(X_train)

# %%
# construct model performance evaluation dataframe
performance_train_dataset = X_train.copy()
performance_train_dataset['y_train'] = y_train
performance_train_dataset['y_hat'] = y_hat

# %%
performance_train_dataset.head()

# %%
confusion_matrix_train = confusion_matrix(y_train, y_hat)
print(confusion_matrix_train)

# %%
train_accuracy_score = accuracy_score(y_train, y_hat)
train_precision_score = precision_score(y_train, y_hat)
train_recall_score = recall_score(y_train, y_hat)
train_f1_score = f1_score(y_train, y_hat)

print('Train accuracy score: {:.3f}'.format(train_accuracy_score))
print('Train precision score: {:.3f}'.format(train_precision_score))
print('Train recall score: {:.3f}'.format(train_recall_score))
print('Train F1 score: {:.3f}'.format(train_f1_score))

# %% [markdown]
# ### Model performance metrics for test dataset

# %%
# Before Testing
# X_test and X_train should be of same data format
# dealing with missing numerical features in X_test using transform method
# keep only thore columns in X_test as that of X_train

# %%
print(X_train.shape)
print(X_test.shape)

# %%
# only keep the columns same as X_train in X_test
X_train_feature_list = X_train.columns.values.tolist()
X_test = X_test[X_train_feature_list]

# %%
print(X_train.shape)
print(X_test.shape)

# %%
X_test_feature_list = list(X_test.select_dtypes(include=['number']).columns)
print(len(X_test_feature_list))
print(X_test_feature_list)

# %%
# now we will rplace the missing values of numeical features with the median value of the feature as in train dataset
Replace_Missing_Numerical_Values(X_test, X_test_feature_list)
# X_test[X_test_feature_list] = imputer_median.transform(X_test[X_test_feature_list])

# %%
X_test.head()

# %%
Check_Missing_Values(X_test)

# %%
y_pred = randomized_search_cv.predict(X_test)

# %%
# construct model performance evaluation dataframe
performance_test_dataset = backup_X_test.copy()
performance_test_dataset['y_test'] = y_test
performance_test_dataset['y_pred'] = y_pred

# %%
performance_test_dataset.head()

# %%
test_confusion_matrix = confusion_matrix(y_test, y_pred)
print(test_confusion_matrix)

# %%
test_accuracy_score = accuracy_score(y_test, y_pred)
test_precision_score = precision_score(y_test, y_pred)
test_recall_score = recall_score(y_test, y_pred)
test_f1_score = f1_score(y_test, y_pred)

print('Test accuracy score: {:.3f}'.format(test_accuracy_score))
print('Test precision score: {:.3f}'.format(test_precision_score))
print('Test recall score: {:.3f}'.format(test_recall_score))
print('Test F1 score: {:.3f}'.format(test_f1_score))

# %% [markdown]
# ### Model evaluation summary

# %%
# difference between nrmse_train and nrmse_test
f1_difference = abs(test_f1_score - train_f1_score)*100/train_f1_score

print('Train accuracy score: {:.3f}'.format(train_accuracy_score))
print('Test accuracy score: {:.3f}'.format(test_accuracy_score))

print('Train F1 score: {:.3f}'.format(train_f1_score))
print('Test F1 score: {:.3f}'.format(test_f1_score))

print('Difference between train_f1_score and test_f1_score: {:.2f} %'.format(f1_difference))

# %%



