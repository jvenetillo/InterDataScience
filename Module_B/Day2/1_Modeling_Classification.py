# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction to Data Science
# MAGIC 
# MAGIC ## Predictive Analysis - numerical and categorical data
# MAGIC 
# MAGIC ### The sinking of Titanic  
# MAGIC Based on [this](https://www.kaggle.com/c/titanic-gettingStarted) Kaggle Competition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Mighty Titanic !
# MAGIC 
# MAGIC ![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1280px-RMS_Titanic_3.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# MAGIC 
# MAGIC One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# MAGIC 
# MAGIC In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# COMMAND ----------

# MAGIC %md
# MAGIC VARIABLE DESCRIPTIONS:
# MAGIC survival        Survival
# MAGIC                 (0 = No; 1 = Yes)
# MAGIC pclass          Passenger Class
# MAGIC                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# MAGIC name            Name
# MAGIC sex             Sex
# MAGIC age             Age
# MAGIC sibsp           Number of Siblings/Spouses Aboard
# MAGIC parch           Number of Parents/Children Aboard
# MAGIC ticket          Ticket Number
# MAGIC fare            Passenger Fare
# MAGIC cabin           Cabin
# MAGIC embarked        Port of Embarkation
# MAGIC                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# MAGIC 
# MAGIC SPECIAL NOTES:
# MAGIC Pclass is a proxy for socio-economic status (SES)
# MAGIC  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# MAGIC 
# MAGIC Age is in Years; Fractional if Age less than One (1)
# MAGIC  If the Age is Estimated, it is in the form xx.5
# MAGIC 
# MAGIC With respect to the family relation variables (i.e. sibsp and parch)
# MAGIC some relations were ignored.  The following are the definitions used
# MAGIC for sibsp and parch.
# MAGIC 
# MAGIC Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# MAGIC Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# MAGIC Parent:   Mother or Father of Passenger Aboard Titanic
# MAGIC Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# MAGIC 
# MAGIC Other family relatives excluded from this study include cousins,
# MAGIC nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# MAGIC only with a nanny, therefore parch=0 for them.  As well, some
# MAGIC travelled with very close friends or neighbors in a village, however,
# MAGIC the definitions do not support such relations.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 - Preparing Environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 - (Installing and) Loading Basic Packages

# COMMAND ----------

#!pip install -U -q  scikit-learn 
#!pip install -U -q  imbalanced-learn 
#!pip install -U -q  xgboost 
#!pip install -U -q  lightgbm 
!pip install -U -q  rgf_python 
#!pip install -U -q catboost
#!pip install -U -q  forestci
!pip install -U -q  tpot 
#!pip install -U -q pycaret
#!pip install -U -q  tensorflow tensorboard 
#!pip install -U -q  torch torchvision 
#!pip install -U -q  delayed
#!pip install -U -q  joblib 

# COMMAND ----------

import os
import sys
import time
import pickle
import itertools
import warnings
import string
import re

import numpy as np
import pylab
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from IPython.display import display, Image
from IPython.core.interactiveshell import InteractiveShell

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 - Testing if GPU is present

# COMMAND ----------

## Optional -- testing GPU support to tensorflow
import tensorflow as tf
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 - Disabling [MLFlow autologging](https://docs.databricks.com/mlflow/databricks-autologging.html)  

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 - Loading Dataset and Distinguishing Attributes

# COMMAND ----------

datapath = "./data/Titanic"

# COMMAND ----------

df_train = pd.read_csv(os.path.join(datapath,'kaggle_titanic_train.csv'))
# df_test = pd.read_csv(os.path.join(datapath,'kaggle_titanic_test.csv')) # No labels, not using

# COMMAND ----------

df_train.head()

# COMMAND ----------

df_train.info(verbose=True, show_counts=True)

# COMMAND ----------

#print(df_train.columns)
print(df_train.select_dtypes(include='number').columns)
print(df_train.select_dtypes(include='object').columns)
print(df_train.select_dtypes(include='category').columns)

# COMMAND ----------

print(df_train.dtypes)
#print(df_train.dtypes[df_train.dtypes.map(lambda x: x =='int64')])
#print(df_train.dtypes[df_train.dtypes.map(lambda x: x =='float64')])
#print(df_train.dtypes[df_train.dtypes.map(lambda x: x =='object')])

# COMMAND ----------

for cat in df_train.columns:
    print(f"Number of levels in category '{cat}': \b {df_train[cat].unique().size:2.2f} ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 - Examining the numeric values

# COMMAND ----------

df_train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 - Examining the categorical values (first 20)

# COMMAND ----------

# There are many values for name and ticket

for cat in df_train.select_dtypes(include='object').columns:
    print(f"Unique values for category '{cat}': \b {df_train[cat].unique()[0:20]}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 - Checking for missing values  
# MAGIC This part demands a closer look to check if there are missing values not immediately identifiable

# COMMAND ----------

df_train.isnull().sum()

# COMMAND ----------

total = df_train.isnull().sum().sort_values(ascending=False)
percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.heatmap(df_train.isnull(),cbar=False,cmap='viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC Using missingno library to explore missing values

# COMMAND ----------

#!pip install missingno
import missingno as msno

# COMMAND ----------

msno.bar(df_train, figsize=(12,6))

# COMMAND ----------

msno.matrix(df_train, figsize=(12,6))

# COMMAND ----------

msno.heatmap(df_train, figsize=(12,6))

# COMMAND ----------

msno.dendrogram(df_train, figsize=(12,6))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 - Data Transformations
# MAGIC  + Creating Train and Test subsets
# MAGIC  + Encoding Categorical Fields (Sex, Embarked, Name, Ticket, Cabin) 
# MAGIC  + Decide how to fill the Missing Values (Embarked, Cabin, Age)  
# MAGIC  + Standardizing Numerical Fields (Age, Pclass, SibSp, Parch, Fare) + Encoded Categorical Fields
# MAGIC  + Encoding Target variable (Survived))¶

# COMMAND ----------

# https://scikit-learn.org/stable/modules/cross_validation.html
from sklearn import model_selection

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
from sklearn.preprocessing import OneHotEncoder

# https://scikit-learn.org/stable/modules/impute.html
from sklearn.impute import SimpleImputer

# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
from sklearn.preprocessing import LabelBinarizer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 - Creating train & test subsets:

# COMMAND ----------

df_train.drop(["PassengerId"], axis=1, inplace=True)
y = df_train.pop("Survived")
y

# COMMAND ----------

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train, y, test_size=0.2, random_state=0)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# COMMAND ----------

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 - Transforming Sex/Gender (stateful/binary)

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train['Sex'].values.reshape(-1, 1))

# COMMAND ----------

enc.categories_[0]

# COMMAND ----------

gender_train = pd.DataFrame(enc.transform(X_train['Sex'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0], dtype=np.int8)
gender_test = pd.DataFrame(enc.transform(X_test['Sex'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0], dtype=np.int8)

# COMMAND ----------

gender_train[0:2]

# COMMAND ----------

X_train.drop(['Sex'], axis=1, inplace=True)
X_train = pd.concat([X_train, gender_train], axis=1)
X_test.drop(['Sex'], axis=1, inplace=True)
X_test = pd.concat([X_test, gender_test], axis=1)
X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 - Transforming Embarked (stateful)  
# MAGIC  We can decide to substitute the missing values using a missing value imputation technique, or simply encoding the missing values as an additional category using OHE

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train['Embarked'].values.reshape(-1, 1))

# COMMAND ----------

embarked_train = pd.DataFrame(enc.transform(X_train['Embarked'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
embarked_train.rename({np.nan: "U"}, axis='columns', inplace=True)   #Rename the column with name "nan" to avoid further problems
embarked_test = pd.DataFrame(enc.transform(X_test['Embarked'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
embarked_test.rename({np.nan: "U"}, axis='columns', inplace=True)

# COMMAND ----------

embarked_train.columns

# COMMAND ----------

embarked_train[0:2]

# COMMAND ----------

X_train.drop(['Embarked'], axis=1, inplace=True)
X_train = pd.concat([X_train, embarked_train], axis=1)
X_test.drop(['Embarked'], axis=1, inplace=True)
X_test = pd.concat([X_test, embarked_test], axis=1)
X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 - Transforming Name (stateful)  
# MAGIC Name is a textual field. We can be very creative when deciding to create features, vectorizing every word, etc.  
# MAGIC In this notebook, we will just extract the titles and surnames and create features out of it.

# COMMAND ----------

# Create function that take name and separates it into title, family name and deletes all puntuation from name column:
def name_sep(data):
    families=[]
    titles = []
    new_name = []
    #for each row in dataset:
    for i in range(len(data)):
        name = data.iloc[i]
        # extract name inside brackets into name_bracket:
        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(",")[0]
        title = name_no_bracket.split(",")[1].strip().split(" ")[0]
        
        #remove punctuations accept brackets:
        for c in string.punctuation:
            name = name.replace(c,"").strip()
            family = family.replace(c,"").strip()
            title = title.replace(c,"").strip()
            
        families.append(family)
        titles.append(title)
        new_name.append(name)
            
    return families, titles, new_name 

# COMMAND ----------

dict_train_name = dict(zip(['surname', 'title', 'newname'], name_sep(X_train["Name"])))
X_train_name = pd.DataFrame(dict_train_name)
dict_test_name = dict(zip(['surname', 'title', 'newname'], name_sep(X_test["Name"])))
X_test_name = pd.DataFrame(dict_test_name)

X_train_name.head()

# COMMAND ----------

X_train_name.title.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's reduce the feature space converting some equivalent titles:

# COMMAND ----------

X_train_name['title'] = X_train_name['title'].replace(['Ms', 'Mlle'],'Miss')
X_train_name['title'] = X_train_name['title'].replace(['Mme'],'Mrs')
X_train_name['title'] = X_train_name['title'].replace(['Dr','Rev','the','Jonkheer','Lady','Sir', 'Don'],'Nobles')
X_train_name['title'] = X_train_name['title'].replace(['Major','Col', 'Capt'],'Navy')

X_test_name['title'] = X_test_name['title'].replace(['Ms', 'Mlle'],'Miss')
X_test_name['title'] = X_test_name['title'].replace(['Mme'],'Mrs')
X_test_name['title'] = X_test_name['title'].replace(['Dr','Rev','the','Jonkheer','Lady','Sir', 'Don'],'Nobles')
X_test_name['title'] = X_test_name['title'].replace(['Major','Col', 'Capt'],'Navy')

# COMMAND ----------

X_train_name.title.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.1 - Encoding Title

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train_name['title'].values.reshape(-1, 1))

# COMMAND ----------

title_train = pd.DataFrame(enc.transform(X_train_name['title'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
title_test = pd.DataFrame(enc.transform(X_test_name['title'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)

# COMMAND ----------

title_train.head(5)

# COMMAND ----------

X_train.drop(['Name'], axis=1, inplace=True)
X_train = pd.concat([X_train, title_train], axis=1)
X_test.drop(['Name'], axis=1, inplace=True)
X_test = pd.concat([X_test, title_test], axis=1)
X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.2 - Encoding Surname
# MAGIC First let's examine the frequency of surnames:

# COMMAND ----------

X_train_name.surname.value_counts().plot(kind="hist", figsize=(12,4))

# COMMAND ----------

# MAGIC %md
# MAGIC Given that most of the surnames are unique, let's just use a frequency encoder

# COMMAND ----------

surnames_train = pd.Series(X_train_name.surname.value_counts().values, name="Surname")
X_train = pd.concat([X_train, surnames_train], axis=1)

surnames_test = pd.Series(X_test_name.surname.value_counts().values, name="Surname")
X_test = pd.concat([X_test, surnames_test], axis=1)

X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 - Transforming Ticket (stateless)

# COMMAND ----------

# MAGIC %md
# MAGIC We can indeed data mine out some interesting relationship from Ticket number. But I think it is best to drop it.  
# MAGIC We could find some relationship between Pclass and Ticket number, and although one could see that there might be some relationship between them, it is not very strong  
# MAGIC We decided to drop this variable as well.

# COMMAND ----------

X_train.drop(['Ticket'], axis=1, inplace=True)
X_test.drop(['Ticket'], axis=1, inplace=True)
X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 - Transforming Cabin (stateful)

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of using the Cabin feature as it is - using traditional hot-encoding, we can try using only the letters.  
# MAGIC After doing some research, we could know that a cabin number looks like ‘C123’ and the letter refers to the deck.  
# MAGIC + We’re going to extract these and create a new feature, that contains a persons deck.  
# MAGIC + And then we will convert the feature into a numeric variable.

# COMMAND ----------

X_train['Cabin'].value_counts()

# COMMAND ----------

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [X_train, X_test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# COMMAND ----------

X_train.drop(['Cabin'], axis=1, inplace=True)
X_test.drop(['Cabin'], axis=1, inplace=True)

X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 - Transforming Age (stateful)  
# MAGIC We will use the iterative inputer that chooses to fill the missing values using a model with the other features

# COMMAND ----------

imp_mean = IterativeImputer(random_state=0)

# COMMAND ----------

imp_mean.fit(X_train)

# COMMAND ----------

X_train = pd.DataFrame(imp_mean.transform(X_train.values), columns=X_train.columns)
X_test = pd.DataFrame(imp_mean.transform(X_test.values), columns=X_test.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8 - Dealing with numerif fields values
# MAGIC + Check there are no more missing values   
# MAGIC + Check all features are numeric
# MAGIC + Standardizing features

# COMMAND ----------

print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())

# COMMAND ----------

X_train.head()

# COMMAND ----------

X_test.head()

# COMMAND ----------

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html  
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler = StandardScaler()

for column in X_train.columns:
    scaler.fit(X_train[column].values.reshape(-1, 1))
    X_train[column] = scaler.transform(X_train[column].values.reshape(-1, 1))
    X_test[column] = scaler.transform(X_test[column].values.reshape(-1, 1))

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.9 - Balancing the classes  
# MAGIC 
# MAGIC As the unbalance is not that big, you can experiment running the algorithms without balancing as well

# COMMAND ----------

print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.9.1 - Oversampling the minority class

# COMMAND ----------

# http://contrib.scikit-learn.org/imbalanced-learn/  
from imblearn.over_sampling import SMOTE

# COMMAND ----------

smote = SMOTE(random_state=0)

X_train, y_train = smote.fit_resample(X_train, y_train)

# COMMAND ----------

print(X_train.shape)
print(y_train.shape)
print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))
print(X_test.shape)
print(y_test.shape)
print(len(y_test[y_test == 0]))
print(len(y_test[y_test == 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 - Metrics and Evaluation Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 - Importing Metrics 

# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 - Creating Cross validation folds   
# MAGIC This will make more precise the evaluation of the classifiers when doing grid search.  
# MAGIC Bear in mind that this also affects the sample used for training, making the performance slighty worse than when using the whole dataset
# MAGIC #https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

# COMMAND ----------

#cv = model_selection.StratifiedKFold(n_splits=10)
cv = model_selection.KFold(n_splits=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 - Creating functions to evaluate the models
# MAGIC 
# MAGIC These functions can be put aside in a python file and imported, to make the code less cluttered

# COMMAND ----------

def mean_scores_cv(clf, cv, X, y):
    scores = model_selection.cross_val_score(clf, X, y, 
                                             scoring=None, 
                                             cv=cv, 
                                             n_jobs=1,
                                             verbose=0,
                                             fit_params=None,
                                             pre_dispatch='2*n_jobs')
    return scores.mean()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
    
def clf_eval(clf, X, y_true, classes=['Perished', 'Survived']):
    y_pred = clf.predict(X)
    clf_matrix = confusion_matrix(y_true, y_pred)
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=classes))
    print('ROC Score: {}'.format(roc_auc_score(y_true, y_pred)))
    print('Accuracy Score: {}'.format(accuracy_score(y_true, y_pred)))
    print('Average Precision Score: {}'.format(average_precision_score(y_true, y_pred)))
    print('f1 Score: {}'.format(f1_score(y_true, y_pred)))
    plot_confusion_matrix(clf_matrix, classes=classes)
    return roc_auc_score(y_true, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5 - Running Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 - Linear Classifiers
# MAGIC 
# MAGIC A linear classifier achieves this by making a classification decision based on the value of a linear combination of the characteristics. An object's characteristics are also known as feature values and are typically presented to the machine in a vector called a feature vector. Such classifiers work well for practical problems such as document classification, and more generally for problems with many variables (features), reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1 - [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
# MAGIC 
# MAGIC Logistic regression is a classification algorithm based on the function which is used at the core of the method, logistic function or sigmoid function. It’s an S-shaped curve that is used to predict a binary outcome (1/0, Yes/No, True/False) given a set of independent variables.
# MAGIC 
# MAGIC + It can also be thought of as a special case of linear regression when the outcome variable is categorical, where we are using the log of odds as a dependent variable.
# MAGIC + It predicts the probability of occurrence of an event by fitting data to a logit function.
# MAGIC 
# MAGIC $f(x) = \frac{L}{1 + e^{-k(x-x_0)}} $  
# MAGIC 
# MAGIC Logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. In the binary case, a dependent variable (the output) can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.
# MAGIC 
# MAGIC 
# MAGIC ![Logistic](https://www.saedsayad.com/images/LogReg_1.png)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_lr = LogisticRegression(penalty='l2',
# MAGIC                             dual=False, 
# MAGIC                             tol=0.001, 
# MAGIC                             C=0.10, 
# MAGIC                             fit_intercept=True, 
# MAGIC                             intercept_scaling=1, 
# MAGIC                             class_weight=None, 
# MAGIC                             random_state=0, 
# MAGIC                             solver='saga', 
# MAGIC                             max_iter=100, 
# MAGIC                             multi_class='ovr', 
# MAGIC                             verbose=0, 
# MAGIC                             warm_start=False, 
# MAGIC                             n_jobs=-1).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_lr = clf_eval(clf_lr, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1.2 - Logistic Regression with Grid Search
# MAGIC 
# MAGIC For each model, we can also run with hyperparameter search for finding the best parameters

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC estimator = LogisticRegression(dual=False, 
# MAGIC                                penalty='l2',
# MAGIC                                fit_intercept=True, 
# MAGIC                                intercept_scaling=1, 
# MAGIC                                class_weight=None, 
# MAGIC                                random_state=0, 
# MAGIC                                solver='saga', 
# MAGIC                                multi_class='ovr', 
# MAGIC                                verbose=0, 
# MAGIC                                warm_start=False, 
# MAGIC                                n_jobs=1)
# MAGIC 
# MAGIC Cs = np.linspace(0,1,10)
# MAGIC max_iter = [10, 50, 100, 200]
# MAGIC tol = [1e-5, 1e-4, 1e-3]
# MAGIC param_grid=dict(C=Cs, max_iter=max_iter)
# MAGIC 
# MAGIC 
# MAGIC clf_lr2 = model_selection.GridSearchCV(param_grid=param_grid,                       ## Grid Search (more exhaustive)
# MAGIC #clf_lr2 = model_selection.RandomizedSearchCV(param_distributions=param_grid,       ## Randomized (faster)
# MAGIC                                        estimator=estimator,
# MAGIC                                        cv=cv,                                       ## using cross validation
# MAGIC                                        n_jobs=1)
# MAGIC clf_lr2.fit(X_train, y_train)
# MAGIC 
# MAGIC print(clf_lr2.best_score_)
# MAGIC print(clf_lr2.best_estimator_.C)
# MAGIC print(clf_lr2.best_estimator_.max_iter)
# MAGIC print(clf_lr2.best_estimator_.tol)

# COMMAND ----------

roc_lr = clf_eval(clf_lr2, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1.3 - Assessing the importance of the features

# COMMAND ----------

coefs = pd.Series(clf_lr.coef_[0], index=X_train.columns)
coefs = coefs.sort_values()
coefs.plot(kind="bar", figsize=(10,6))
print(coefs.sort_values(ascending = True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1.4 -  Precision Recall Curve
# MAGIC 
# MAGIC This graph can be used for all classifiers that expose the probabilities of prediction for the classes:  
# MAGIC For each person the classifier algorithm has to classify, it computes a probability based on a function  
# MAGIC It classifies the person as survived (when the score is bigger than the threshold) or as not survived (when the score is smaller than the threshold).  
# MAGIC That’s why the threshold plays an important part.  
# MAGIC 
# MAGIC We will plot the precision and recall while varying the threshold:

# COMMAND ----------

# getting the probabilities of our predictions
y_scores = clf_lr.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Another way to find the best balance is to plot the precision and recall against each other:

# COMMAND ----------

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precision, recall)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.2 - [Ridge Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)  
# MAGIC 
# MAGIC Ridge Classifiers addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients.  
# MAGIC The ridge coefficients minimize a penalized residual sum of squares  
# MAGIC https://stats.stackexchange.com/questions/558900/why-ridgeclassifier-can-be-significantly-faster-than-logisticregression-with-a-h

# COMMAND ----------

from sklearn.linear_model import RidgeClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_rdg = RidgeClassifier(alpha=1.0, 
# MAGIC                           fit_intercept=True, 
# MAGIC                           normalize=False, 
# MAGIC                           copy_X=True, 
# MAGIC                           max_iter=None, 
# MAGIC                           tol=0.001, 
# MAGIC                           class_weight=None, 
# MAGIC                           solver='auto', 
# MAGIC                           random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_rdg = clf_eval(clf_rdg, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.3 - [Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)
# MAGIC 
# MAGIC Perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time.

# COMMAND ----------

from sklearn.linear_model import Perceptron

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_pcp = Perceptron(penalty=None,
# MAGIC                      alpha=0.001,
# MAGIC                      fit_intercept=True,
# MAGIC                      max_iter=230,
# MAGIC                      shuffle=True,
# MAGIC                      verbose=0,
# MAGIC                      eta0=1.0,
# MAGIC                      n_jobs=-1, 
# MAGIC                      random_state=0, 
# MAGIC                      class_weight=None, 
# MAGIC                      warm_start=False).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_pcp = clf_eval(clf_pcp, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.4 - [Passive Aggressive](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)  
# MAGIC 
# MAGIC https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/ 
# MAGIC 
# MAGIC Passive Aggressive Algorithms are a family of online learning algorithms (for both classification and regression) proposed by Crammer at al. The idea is very simple and their performance has been proofed to be superior to many other alternative methods like Online Perceptron and MIRA.  
# MAGIC Passive: if correct classification, keep the model; Aggressive: if incorrect classification, update to adjust to this misclassified example. 
# MAGIC In passive, the information hidden in the example is not enough for updating; in aggressive, the information shows that at lest this time you are wrong, a better model should modify this mistake.

# COMMAND ----------

from sklearn.linear_model import PassiveAggressiveClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_pac = PassiveAggressiveClassifier(C=0.1, 
# MAGIC                                       fit_intercept=True, 
# MAGIC                                       max_iter=100, 
# MAGIC                                       shuffle=True, 
# MAGIC                                       verbose=0, 
# MAGIC                                       loss='hinge', 
# MAGIC                                       n_jobs=-1, 
# MAGIC                                       random_state=0, 
# MAGIC                                       warm_start=False, 
# MAGIC                                       class_weight=None).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_pac = clf_eval(clf_pac, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.5 - [SGDC Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
# MAGIC 
# MAGIC The SGDC classifier is a linear classifier (SVM, logistic regression) with SGD training. Implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.  
# MAGIC 
# MAGIC The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.

# COMMAND ----------

from sklearn.linear_model import SGDClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_sgdc = SGDClassifier(loss='hinge',
# MAGIC                          penalty='l2', 
# MAGIC                          alpha=0.0001,
# MAGIC                          l1_ratio=0.15, 
# MAGIC                          fit_intercept=True,
# MAGIC                          max_iter=200, 
# MAGIC                          shuffle=True,
# MAGIC                          verbose=0,
# MAGIC                          epsilon=0.01,
# MAGIC                          n_jobs=-1,
# MAGIC                          random_state=0,
# MAGIC                          learning_rate='optimal',
# MAGIC                          eta0=0.0, 
# MAGIC                          power_t=0.5,
# MAGIC                          class_weight=None,
# MAGIC                          warm_start=False, 
# MAGIC                          average=False).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_sgdc = clf_eval(clf_sgdc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.6 - [Support Vector Machines](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/  
# MAGIC 
# MAGIC Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# MAGIC 
# MAGIC In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
# MAGIC 
# MAGIC An SVM will find a hyperplane or a boundary between the two classes of data that maximizes. There are other planes as well which can separate the two classes, but only the SVM hyperplane can maximize the margin between the classes.
# MAGIC 
# MAGIC B0 + (B1 * X1) + (B2 * X2) = 0 where, B1 and B2 determines the slope of the line and B0 (intercept) found by the learning algorithm. X1 and X2 are the two input variables.
# MAGIC 
# MAGIC ![SVM](https://www.researchgate.net/publication/331308937/figure/fig1/AS:870140602249216@1584469094771/Illustration-of-support-vector-machine-SVM-to-generalize-the-optimal-separating.ppm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Choosing the best parameters using [GridSearchCV](http://scikit-learn.org/stable/modules/grid_search.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

# COMMAND ----------

from sklearn import svm

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC estimator = svm.SVC()
# MAGIC 
# MAGIC kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# MAGIC Cs = np.linspace(0.1,3,7)
# MAGIC degrees = [2,3,4,5]
# MAGIC gammas = np.logspace(-5, 0, 7)
# MAGIC 
# MAGIC param_grid=dict(kernel=kernels, C=Cs, gamma=gammas, degree=degrees)
# MAGIC 
# MAGIC 
# MAGIC clf_svc = model_selection.GridSearchCV(param_grid=param_grid,                       ## Grid Search (more exhaustive)
# MAGIC #clf_svc = model_selection.RandomizedSearchCV(param_distributions=param_grid,       ## Randomized (faster)
# MAGIC                                        estimator=estimator,
# MAGIC                                        cv=cv,                                       ## using cross validation
# MAGIC                                        n_jobs=-1)
# MAGIC 
# MAGIC clf_svc.fit(X_train, y_train)
# MAGIC 
# MAGIC print(clf_svc.best_score_)
# MAGIC print(clf_svc.best_estimator_.kernel)
# MAGIC print(clf_svc.best_estimator_.C)
# MAGIC print(clf_svc.best_estimator_.degree)
# MAGIC print(clf_svc.best_estimator_.gamma)

# COMMAND ----------

roc_svc = clf_eval(clf_svc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 - [k-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)  
# MAGIC 
# MAGIC k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# MAGIC 
# MAGIC + In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# MAGIC 
# MAGIC + In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.
# MAGIC 
# MAGIC k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.
# MAGIC 
# MAGIC It works by finding the distances between the new data point added and the points already existed in the two separate classes. Whatever class got the highest votes, the new data point belongs to that class.
# MAGIC 
# MAGIC EuclideanDistance(x, xi) = sqrt( sum( (xj — xij)² ) )
# MAGIC 
# MAGIC ![knn](https://miro.medium.com/max/1400/1*T8Pnw0kiVbrPGnqnB2I_Zw.jpeg)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_knn = KNeighborsClassifier(n_neighbors=25,
# MAGIC                                weights='uniform', 
# MAGIC                                algorithm='auto', 
# MAGIC                                leaf_size=30, 
# MAGIC                                p=4, 
# MAGIC                                metric='minkowski', 
# MAGIC                                metric_params=None, 
# MAGIC                                n_jobs=-1).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_knn = clf_eval(clf_knn, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 - [Decision Trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
# MAGIC 
# MAGIC Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# MAGIC 
# MAGIC Decision tree algorithms are referred to as CART or Classification and Regression Trees. It is a flowchart like a tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
# MAGIC 
# MAGIC + A Gini score gives an idea of how good a split is by how mixed the response classes are in the groups created by the split.
# MAGIC 
# MAGIC ![dtree](https://elf11.github.io/images/decisionTree.png)

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_dtc = DecisionTreeClassifier(criterion='gini', 
# MAGIC                                  splitter='best', 
# MAGIC                                  max_depth=None, 
# MAGIC                                  min_samples_split=3, 
# MAGIC                                  min_samples_leaf=1, 
# MAGIC                                  min_weight_fraction_leaf=0.0, 
# MAGIC                                  max_features=None, 
# MAGIC                                  random_state=0, 
# MAGIC                                  max_leaf_nodes=None, 
# MAGIC                                  class_weight=None,).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_dtc = clf_eval(clf_dtc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 - [Ensemble Classifiers](http://scikit-learn.org/stable/modules/ensemble.html)
# MAGIC 
# MAGIC A linear classifier achieves this by making a classification decision based on the value of a linear combination of the characteristics. An object's characteristics are also known as feature values and are typically presented to the machine in a vector called a feature vector. Such classifiers work well for practical problems such as document classification, and more generally for problems with many variables (features), reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.
# MAGIC 
# MAGIC We have many type of ensembles: 
# MAGIC + Bagging
# MAGIC + Boosting 
# MAGIC + Voting & Stacking

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4.1 - Bagging
# MAGIC 
# MAGIC Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach. Random Forests are the most common type of bagging algorithms.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.1.1 -  [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
# MAGIC 
# MAGIC http://blog.yhathq.com/posts/random-forests-in-python.html  
# MAGIC http://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/  
# MAGIC 
# MAGIC Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# MAGIC 
# MAGIC + Random forests involve creating multiple decision trees using bootstrapped datasets of the original data and randomly selecting a subset of variables at each step of the decision tree.     
# MAGIC + The model then selects the mode of all of the predictions of each decision tree (bagging).

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_rf = RandomForestClassifier(n_estimators=300, 
# MAGIC                                 criterion='gini', 
# MAGIC                                 max_depth=None, 
# MAGIC                                 min_samples_split=3, #2,
# MAGIC                                 min_samples_leaf=1, 
# MAGIC                                 min_weight_fraction_leaf=0.0, 
# MAGIC                                 max_features='sqrt', 
# MAGIC                                 max_leaf_nodes=None, 
# MAGIC                                 bootstrap=True, 
# MAGIC                                 oob_score=False, 
# MAGIC                                 n_jobs=-1, 
# MAGIC                                 random_state=0, 
# MAGIC                                 verbose=0, 
# MAGIC                                 warm_start=False, 
# MAGIC                                 class_weight=None).fit(X_train, y_train)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC roc_rf = clf_eval(clf_rf, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assessing the importance of the features

# COMMAND ----------

features = X_train.columns
importances = clf_rf.feature_importances_
indices = np.argsort(importances) #[0:9])  # top 10 features
plt.figure(figsize=(12,10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.1.2 -  [Bagging Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)   
# MAGIC 
# MAGIC A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

# COMMAND ----------

from sklearn.ensemble import BaggingClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_bgc = BaggingClassifier(clf_lr).fit(X_train, y_train)   ## Using the previously fitted Logistic Regression
# MAGIC roc_bgc = clf_eval(clf_bgc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.1.3 - [Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

# COMMAND ----------

from sklearn.ensemble import ExtraTreesClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_etc = ExtraTreesClassifier(n_estimators=300,
# MAGIC                                max_depth=None,
# MAGIC                                min_samples_split=3,
# MAGIC                                random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_etc = clf_eval(clf_etc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4.2 - Boosting
# MAGIC 
# MAGIC Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones. Boosting is based on the question posed by Kearns and Valiant (1988, 1989): Can a set of weak learners create a single strong learner? A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.1 - [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
# MAGIC 
# MAGIC Adaptive boost is also an ensemble algorithm that leverages bagging and boosting methods to develop an enhanced predictor.
# MAGIC 
# MAGIC + AdaBoost creates a forest of stumps rather than trees. A stump is a tree that is made of only one node and two leaves.
# MAGIC + AdaBoost takes a more iterative approach in the sense that it seeks to iteratively improve from the mistakes that the previous stump(s) made.

# COMMAND ----------

from sklearn.ensemble import AdaBoostClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_abc = AdaBoostClassifier(base_estimator=None,
# MAGIC                              n_estimators=300,
# MAGIC                              learning_rate=0.1,
# MAGIC                              algorithm='SAMME.R',
# MAGIC                              random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_abc = clf_eval(clf_abc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.2 - [Gradient Boost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html):  
# MAGIC 
# MAGIC Gradient Boost is also an ensemble algorithm that uses boosting methods to develop an enhanced predictor.
# MAGIC 
# MAGIC + Unlike AdaBoost which builds stumps, Gradient Boost builds trees with usually 8–32 leaves.
# MAGIC + Gradient Boost views the boosting problem as an optimization problem, where it uses a loss function and tries to minimize the error. This is why it’s called Gradient boost, as it’s inspired by gradient descent.

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier

# COMMAND ----------

# MAGIC %time
# MAGIC 
# MAGIC clf_gbc = GradientBoostingClassifier(learning_rate=0.1,
# MAGIC                                      n_estimators=100,
# MAGIC                                      subsample=1.0,
# MAGIC                                      criterion='friedman_mse',
# MAGIC                                      min_samples_split=3, 
# MAGIC                                      min_samples_leaf=1, 
# MAGIC                                      min_weight_fraction_leaf=0.0, 
# MAGIC                                      max_depth=3, 
# MAGIC                                      init=None, 
# MAGIC                                      random_state=0, 
# MAGIC                                      max_features=None, 
# MAGIC                                      verbose=0, 
# MAGIC                                      max_leaf_nodes=None, 
# MAGIC                                      warm_start=False,
# MAGIC                                      validation_fraction=0.1, 
# MAGIC                                      n_iter_no_change=None, 
# MAGIC                                      tol=0.0001, 
# MAGIC                                      ccp_alpha=0.0).fit(X_train, y_train)
# MAGIC                                      
# MAGIC roc_gbc = clf_eval(clf_gbc, X_test, y_test)                                     

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.3 - [XGBoost](https://github.com/dmlc/xgboost/tree/master/python-package)  
# MAGIC 
# MAGIC XGBoost is one of the most popular and widely used algorithms today because it is simply so powerful.
# MAGIC 
# MAGIC + It is similar to Gradient Boost but has a few extra features that make it that much stronger.
# MAGIC + Newton Boosting — Provides a direct route to the minima than gradient descent, making it much faster.
# MAGIC + An extra randomization parameter — reduces the correlation between trees, ultimately improving the strength of the ensemble.

# COMMAND ----------

import xgboost

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_xgb = xgboost.sklearn.XGBClassifier(base_score=0.5,
# MAGIC                                         learning_rate=0.1,
# MAGIC                                         n_estimators=250,
# MAGIC                                         max_delta_step=0,
# MAGIC                                         max_depth=2,
# MAGIC                                         min_child_weight=1,
# MAGIC                                         missing=1,
# MAGIC                                         gamma=0.1,
# MAGIC                                         subsample=1,
# MAGIC                                         colsample_bylevel=1,
# MAGIC                                         colsample_bytree=1,
# MAGIC                                         objective= 'binary:logitraw',
# MAGIC                                         #objective='multi:softprob',
# MAGIC                                         eval_metric='auc',
# MAGIC                                         #eval_metric='logloss',
# MAGIC                                         reg_alpha=0, 
# MAGIC                                         reg_lambda=1,
# MAGIC                                         nthread=-1,
# MAGIC                                         scale_pos_weight=1,
# MAGIC                                         seed=0,
# MAGIC                                         use_label_encoder=False,
# MAGIC                                         random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_xgb = clf_eval(clf_xgb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.4 - Light GBM: 
# MAGIC 
# MAGIC It is another type of boosting algorithm that has shown to be faster and sometimes more accurate than XGBoost.
# MAGIC 
# MAGIC + It uses a unique technique called Gradient-based One-side sampling (GOSS) to filter out the data instances to find a split value.

# COMMAND ----------

import lightgbm as lgb

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC params = {'boosting_type': 'gbdt',
# MAGIC           'max_depth' : -1,
# MAGIC           'objective': 'binary',
# MAGIC           'num_leaves': 64, 
# MAGIC           'learning_rate': 0.05, 
# MAGIC           'max_bin': 512, 
# MAGIC           'subsample_for_bin': 200,
# MAGIC           'subsample': 1, 
# MAGIC           'subsample_freq': 1, 
# MAGIC           'colsample_bytree': 0.8, 
# MAGIC           'reg_alpha': 5, 
# MAGIC           'reg_lambda': 10,
# MAGIC           'min_split_gain': 0.15, 
# MAGIC           'min_child_weight': 1, 
# MAGIC           'min_child_samples': 5, 
# MAGIC           'scale_pos_weight': 1,
# MAGIC           'num_class' : 1,
# MAGIC           'metric' : 'binary_error'}
# MAGIC 
# MAGIC lgb_train = lgb.Dataset(X_train, y_train)
# MAGIC lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# MAGIC 
# MAGIC clf_lgb = lgb.LGBMClassifier(boosting_type= 'gbdt',
# MAGIC                              objective = 'binary',
# MAGIC                              n_jobs = -1,
# MAGIC                              max_depth = params['max_depth'],
# MAGIC                              max_bin = params['max_bin'],
# MAGIC                              subsample_for_bin = params['subsample_for_bin'],
# MAGIC                              subsample = params['subsample'],
# MAGIC                              subsample_freq = params['subsample_freq'],
# MAGIC                              min_split_gain = params['min_split_gain'], 
# MAGIC                              min_child_weight = params['min_child_weight'], 
# MAGIC                              min_child_samples = params['min_child_samples'], 
# MAGIC                              scale_pos_weight = params['scale_pos_weight'])
# MAGIC 
# MAGIC clf_lgb.fit(X_train, y_train)
# MAGIC 
# MAGIC 
# MAGIC roc_lgb = clf_eval(clf_lgb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.5 - [Catboost](https://catboost.ai/)  
# MAGIC 
# MAGIC TBD - There is a problem running inside Databricks environment

# COMMAND ----------

#from catboost import CatBoostClassifier

# COMMAND ----------

#tmp_path = '/dbfs/tmp'
#
#clf_ctb = CatBoostClassifier(iterations=2,
#                             learning_rate=1,
#                             depth=2,
#                             train_dir=tmp_path)
#clf_ctb.fit(X_train, y_train, verbose=False)
#roc_ctb = clf_eval(clf_ctb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.6 - [Regularized Greedy Forest](https://github.com/RGF-team/rgf)
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/  
# MAGIC 
# MAGIC In Boosting algorithms, each classifier/regressor is trained on data, taking into account the previous classifiers’/regressors’ success. After each training step, the weights are redistributed. Mis-classified data increases its weights to emphasize the most difficult cases. In this way, subsequent learners will focus on them during their training. However, the boosting methods simply treat the decision tree base learner as a black box and it does not take advantage of the tree structure itself.  In a sense, boosting does a partial corrective step to the model at each iteration. In contrast, RGF performs 2 steps:
# MAGIC 
# MAGIC + Finds the one step structural change to the current forest to obtain the new forest that minimises the loss function (e.g. Least squares or logloss)
# MAGIC + Adjusts the leaf weights for the entire forest to minimize the loss function

# COMMAND ----------

from rgf.sklearn import RGFClassifier, FastRGFClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_rgf = RGFClassifier(max_leaf=100,
# MAGIC                         algorithm="RGF_Sib",
# MAGIC                         test_interval=60,
# MAGIC                         verbose=False,).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_rgf = clf_eval(clf_rgf, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 - [Bayesian Classifiers](http://scikit-learn.org/stable/modules/naive_bayes.html)  
# MAGIC 
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html  
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html  
# MAGIC 
# MAGIC Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression, which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.
# MAGIC 
# MAGIC + As the name specifies, this algorithm is entirely based on Bayes's theorem. Bayes’ theorem says we can calculate the probability of a piece of data belonging to a given class if prior knowledge is given.  
# MAGIC + P(class|data) = (P(data|class) * P(class)) / P(data)

# COMMAND ----------

from sklearn.naive_bayes import BernoulliNB

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_bnb = BernoulliNB(alpha=0.20, 
# MAGIC                      binarize=0.0, 
# MAGIC                      fit_prior=True, 
# MAGIC                      class_prior=None).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_bnb = clf_eval(clf_bnb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 - Gaussian Processes
# MAGIC 
# MAGIC In the simple linear regression setting, we have a dependent variable y that we assume can be modeled as a function of an independent variable x, i.e. y=f(x)+ϵ (where ϵ is the irreducible error) but we assume further that the function f defines a linear relationship and so we are trying to find the parameters $θ_0$ and $θ_1$ which define the intercept and slope of the line respectively, i.e. $y=θ_0+θ_1x+ϵ$.
# MAGIC + Bayesian linear regression provides a probabilistic approach to this by finding a distribution over the parameters that gets updated whenever new data points are observed. 
# MAGIC + The GP approach, in contrast, is a non-parametric approach, in that it finds a distribution over the possible functions $f(x)$ that are consistent with the observed data. As with all Bayesian methods it begins with a prior distribution and updates this as data points are observed, producing the posterior distribution over functions.

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_gpc = GaussianProcessClassifier(kernel=None, 
# MAGIC                                     optimizer='fmin_l_bfgs_b', 
# MAGIC                                     n_restarts_optimizer=0, 
# MAGIC                                     max_iter_predict=100, 
# MAGIC                                     warm_start=False, 
# MAGIC                                     copy_X_train=True, 
# MAGIC                                     random_state=0, 
# MAGIC                                     multi_class='one_vs_rest', 
# MAGIC                                     n_jobs=-1).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_gpc = clf_eval(clf_gpc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.7 - Neural Networks
# MAGIC 
# MAGIC Neural Networks take inspiration from the learning process occurring in human brains. They consists of an artificial network of functions, called parameters, which allows the computer to learn, and to fine tune itself, by analyzing new data. Each "neuron" learns the parameters of a function which produces an output, after receiving one or multiple inputs. Those outputs are then passed to the next layer of neurons, which use them as inputs of their own function, and produce further outputs. Those outputs are then passed on to the next layer of neurons, and so it continues until every layer of neurons have been considered, and the terminal neurons have received their input. Those terminal neurons then output the final result for the model.  

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# MAGIC 
# MAGIC batch_size = int(len(X_train)/25)
# MAGIC num_classes = 2
# MAGIC epochs = 30
# MAGIC np.random.seed(0)
# MAGIC 
# MAGIC model = tf.keras.Sequential()
# MAGIC model.add(tf.keras.layers.Dense(input_dim=X_train.shape[1], units=20, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=40, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=20, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# MAGIC model.summary()
# MAGIC 
# MAGIC model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# MAGIC 
# MAGIC history = model.fit(X_train, y_train,
# MAGIC                     batch_size=batch_size,
# MAGIC                     epochs=epochs,
# MAGIC                     callbacks=[callback],
# MAGIC                     verbose=0,
# MAGIC                     shuffle=False,
# MAGIC                     validation_data=(X_test, y_test))
# MAGIC 
# MAGIC score = model.evaluate(X_test, y_test, verbose=0)
# MAGIC print('Test loss:', score[0])
# MAGIC print('Test accuracy:', score[1])
# MAGIC 
# MAGIC y_pred = (model.predict(X_test) > 0.5).astype("int32")
# MAGIC 
# MAGIC clf_matrix = confusion_matrix(y_test, y_pred)
# MAGIC print('Classification Report')
# MAGIC print(classification_report(y_test, y_pred, target_names=['Perished', 'Survived']))
# MAGIC print('ROC Score: {}'.format(roc_auc_score(y_test, y_pred)))
# MAGIC print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))
# MAGIC print('Average Precision Score: {}'.format(average_precision_score(y_test, y_pred)))
# MAGIC print('f1 Score: {}'.format(f1_score(y_test, y_pred)))
# MAGIC plot_confusion_matrix(clf_matrix, classes=['Perished', 'Survived'])
# MAGIC roc_keras = roc_auc_score(y_test, y_pred)

# COMMAND ----------

# list all data in history
print(history.history.keys())

# COMMAND ----------

# summarize history for loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# summarize history for accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.8 - AutoML / Genetic Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.8.1 - [TPOT](https://github.com/rhiever/tpot)  
# MAGIC 
# MAGIC TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

# COMMAND ----------

from tpot import TPOTClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_tpot = TPOTClassifier(verbosity=1, 
# MAGIC                           max_time_mins=60, 
# MAGIC                           max_eval_time_mins=10, 
# MAGIC                           population_size=100,
# MAGIC                           generations=100,
# MAGIC                           n_jobs=-1)
# MAGIC 
# MAGIC clf_tpot.fit(X_train, y_train)
# MAGIC roc_tpot = clf_eval(clf_tpot, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.8.2 - [PyCaret](https://pycaret.gitbook.io/docs/)  
# MAGIC 
# MAGIC https://pycaret.gitbook.io/docs/get-started/quickstart  
# MAGIC PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows.  
# MAGIC It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.

# COMMAND ----------

#from pycaret.classification import *
#s = setup(pd.concat([X_train, y_train], axis=1), target="Survived")

#best = compare_models()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4.3 - [Voting / Stacking](http://scikit-learn.org/stable/modules/ensemble.html#votingclassifier)  
# MAGIC 
# MAGIC https://medium.com/@satyam-kumar/use-voting-classifier-to-improve-the-performance-of-your-ml-model-805345f9de0e
# MAGIC 
# MAGIC Voting algorithms are simple strategies, where you aglomerate results of classifiers' decisions by for example taking the class which appears in most cases. 
# MAGIC 
# MAGIC Stacking/grading strategies are generalizations of this concept. Instead of simply saying "ok, I have a scheme v, which I will use to select the best answer among my k classifiers" you create another abstraction layer, where you actually learn to predict the correct label having k votes.

# COMMAND ----------

from sklearn.ensemble import VotingClassifier

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC with warnings.catch_warnings():
# MAGIC     warnings.simplefilter("ignore", category=RuntimeWarning)
# MAGIC     ensemble = VotingClassifier(estimators=[('clf_sgdc', clf_sgdc),
# MAGIC                                             ('clf_lgr', clf_lr),
# MAGIC                                             ('clf_rdg', clf_rgf),
# MAGIC                                             ('clf_bgc', clf_bgc),
# MAGIC                                             ('clf_etc', clf_etc),
# MAGIC                                             ('clf_abc', clf_abc),
# MAGIC                                             ('clf_pct', clf_pcp),
# MAGIC                                             ('clf_xgb', clf_xgb),
# MAGIC                                             ('clf_rf', clf_rf),
# MAGIC                                             ('clf_knn', clf_knn),
# MAGIC                                             ('clf_rgf', clf_rgf),
# MAGIC                                             #('clf_tpot', clf_tpot),
# MAGIC                                             ],
# MAGIC                                 voting='hard',
# MAGIC                                 weights=[1,1,1,1,1,1,1,2,1,2,1]).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_ens = clf_eval(ensemble, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6 - Evaluating the results:

# COMMAND ----------

dic_results = {'SVM': roc_svc,
               'RandomForest': roc_rf,
               'DecisionTree': roc_dtc,
               'ExtraTree': roc_etc,
               'AdaBoost': roc_abc,
               'GradBoost': roc_gbc,
               'LightGBM': roc_lgb,
               'SGDC': roc_sgdc,
               'Ridge': roc_rdg,
               'Perceptron': roc_pcp,
               'PassAgre': roc_pac,
               'LogiReg': roc_lr,
               'BernouNB': roc_bnb,
               'RGF': roc_rgf,
               'XGBoost':roc_xgb,
               'Knn':roc_knn,
               'Tensorflow': roc_keras,
               'Bagging': roc_bgc,
               'Voting': roc_ens,
               'Tpot': roc_tpot,
               'RGF': roc_rgf,
               'GaussianP': roc_gpc,
              }

import operator
tup_results = sorted(dic_results.items(), key=operator.itemgetter(1))

N = len(dic_results)
ind = np.arange(N)  # the x locations for the groups
width = 0.40       # the width of the bars

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
rects = ax.bar(ind+0.5, list(zip(*tup_results))[1], width,)
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2., 
            1.005*height, 
            '{0:.4f}'.format(height), 
            ha='center', 
            va='bottom',)

ax.set_ylabel('Scores')
ax.set_ylim(ymin=0.65,ymax = 0.88)
ax.set_title("Classificators' performance")
ax.set_xticks(ind + width/2.)
ax.set_xticklabels(list(zip(*tup_results))[0], rotation=45)

plt.show()

# COMMAND ----------

features = X_train.columns
df_fi = pd.DataFrame({'clf_lr': (abs(clf_lr.coef_[0])/sum(abs(clf_lr.coef_[0]))),    #a sort of adaptation
                      'clf_rf':clf_rf.feature_importances_,
                      'clf_xgb':clf_xgb.feature_importances_,
                      'clf_etc':clf_etc.feature_importances_,
                      'clf_abc':clf_abc.feature_importances_,
                      #'clf_bgc':clf_bgc.estimators_[0].feature_importances_,
                      'clf_bgc':(abs(clf_lr.coef_[0])/sum(abs(clf_lr.coef_[0]))),   # Adapting because we are using linear regression classifiers in our bagging strategy
                      'clf_gbc':clf_gbc.feature_importances_,
                      'clf_lgb':clf_lgb.feature_importances_ / sum(clf_lgb.feature_importances_),
                     },
                      index=features)
df_fi['mean_importance'] = df_fi.mean(axis=1)
df_fi.sort_values(['mean_importance'], ascending=False, inplace=True)
df_fi

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 - Optimizing the best Model

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC opt_model = RandomForestClassifier(min_weight_fraction_leaf=0.0, 
# MAGIC                                    max_features='sqrt', 
# MAGIC                                    max_leaf_nodes=None, 
# MAGIC                                    bootstrap=True, 
# MAGIC                                    oob_score=False, 
# MAGIC                                    n_jobs=-1, 
# MAGIC                                    random_state=0, 
# MAGIC                                    verbose=0, 
# MAGIC                                    warm_start=False, 
# MAGIC                                    class_weight=None)
# MAGIC 
# MAGIC 
# MAGIC n_estimators = [100, 150, 200]
# MAGIC max_depth = [5, 10, None]
# MAGIC min_samples_split = [1,2,3]
# MAGIC min_samples_leaf = [2,3,4]
# MAGIC criterion=['gini', 'entropy', 'log_loss']
# MAGIC param_grid=dict(n_estimators=n_estimators, 
# MAGIC                 max_depth=max_depth, 
# MAGIC                 min_samples_split=min_samples_split, 
# MAGIC                 min_samples_leaf=min_samples_leaf,
# MAGIC                 criterion=criterion)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC clf_rfo = model_selection.GridSearchCV(param_grid=param_grid,                       ## Grid Search (more exhaustive)
# MAGIC #clf_rfo = model_selection.RandomizedSearchCV(param_distributions=param_grid,       ## Randomized (faster)
# MAGIC                                        estimator=opt_model,
# MAGIC                                        cv=cv,                                       ## using cross validation
# MAGIC                                        n_jobs=-1)
# MAGIC 
# MAGIC clf_rfo.fit(X_train, y_train)
# MAGIC 
# MAGIC print(clf_rfo.best_score_)
# MAGIC print(clf_rfo.best_estimator_.n_estimators)
# MAGIC print(clf_rfo.best_estimator_.max_depth)
# MAGIC print(clf_rfo.best_estimator_.min_samples_split)
# MAGIC print(clf_rfo.best_estimator_.min_samples_leaf)
# MAGIC print(clf_rfo.best_estimator_.criterion)

# COMMAND ----------

roc_rfo = clf_eval(clf_rfo, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 - Examining the ROC Curve

# COMMAND ----------

# calculate the fpr and tpr for all thresholds of the classification
probs = clf_rfo.predict_proba(X_train)
y_hat = probs[:,1]
fpr, tpr, threshold = roc_curve(y_train, y_hat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 - Displaying the [learning curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)  
# MAGIC A cross-validation generator splits the whole dataset k times in training and test data. Subsets of the training set with varying sizes will be used to train the estimator and a score for each training subset size and the test set will be computed. Afterwards, the scores will be averaged over all k runs for each training subset size.

# COMMAND ----------

def plot_learning_curve(estimator, 
                        title, 
                        X, 
                        y, 
                        ylim=None, 
                        cv=None,
                        n_jobs=-1, 
                        train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(16,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator,
                                                                            X, 
                                                                            y, 
                                                                            cv=cv,
                                                                            n_jobs=n_jobs,
                                                                            train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, 
                     color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, 
                     color="g")
    
    plt.plot(train_sizes, 
             train_scores_mean, 
             'o-', 
             color="r", 
             label="Training score")
    
    plt.plot(train_sizes, 
             test_scores_mean, 
             'o-', 
             color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# COMMAND ----------

# MAGIC %%time
# MAGIC title = f'''Learning Curves (Random Forest, 
# MAGIC Estimators:{clf_rfo.best_estimator_.n_estimators},
# MAGIC Max Depth:{clf_rfo.best_estimator_.max_depth},
# MAGIC Min Samples Split:{clf_rfo.best_estimator_.min_samples_split},
# MAGIC Min Samples Leaf:{clf_rfo.best_estimator_.min_samples_leaf},
# MAGIC Criterion:{clf_rfo.best_estimator_.criterion})'''
# MAGIC graph = plot_learning_curve(clf_rfo, title, X_train, y_train, cv=cv)
# MAGIC graph.show()
