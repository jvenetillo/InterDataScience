# Databricks notebook source
# MAGIC %md
# MAGIC # Classification Models
# MAGIC 
# MAGIC This notebook will guide you from beginning to end of a classification use case. It will demonstrate several different classification models. 
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
# MAGIC **The Data Set**   
# MAGIC The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# MAGIC 
# MAGIC One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# MAGIC 
# MAGIC In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Varbiable Descriptions
# MAGIC + **survival**: Survival
# MAGIC  (0 = Did not survive; 1 = Survived)
# MAGIC + **pclass**: Passenger Class
# MAGIC  (1 = 1st; 2 = 2nd; 3 = 3rd)
# MAGIC + **name**: Full name of passenger
# MAGIC + **sex**: Sex of passenger
# MAGIC + **age**: Age
# MAGIC + **sibsp**: Number of Siblings/Spouses Aboard
# MAGIC + **parch**: Number of Parents/Children Aboard
# MAGIC + **ticket**: Ticket Number
# MAGIC + **fare**: Passenger Fare
# MAGIC + **cabin**: Cabin
# MAGIC + **embarked**: Port of Embarkation
# MAGIC                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### SPECIAL NOTES:
# MAGIC + **Pclass** is a proxy for socio-economic status (SES):
# MAGIC 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# MAGIC 
# MAGIC + **Age** is in Years; Fractional if the age is less than One (1)   
# MAGIC If the age is estimated, it is in the form xx.5
# MAGIC 
# MAGIC + For the family relation variables (i.e. **sibsp** and **parch**)
# MAGIC some relations were ignored.  The following are the definitions used
# MAGIC for **sibsp** and **parch**:    
# MAGIC **Sibling**:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic   
# MAGIC **Spouse**:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances ignored)   
# MAGIC **Parent**:   Mother or Father of Passenger Aboard Titanic   
# MAGIC **Child**:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic   
# MAGIC Family relatives excluded from this study include cousins,
# MAGIC nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# MAGIC only with a nanny, therefore parch is 0 for them.  Some
# MAGIC travelled with very close friends or neighbors in a village, however,
# MAGIC the definitions do not support such relations.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 - Preparing Environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 - (Installing and) Loading Basic Packages
# MAGIC 
# MAGIC The following packages are needed to run this notebook. Should you find some of them missing when going through, de-comment the respective line to install the package. 

# COMMAND ----------

#!pip install -U -q scikit-learn 
#!pip install -U -q imbalanced-learn 
#!pip install -U -q xgboost 
#!pip install -U -q lightgbm 
!pip install -U -q rgf_python 
!pip install -U -q catboost
#!pip install -U -q forestci
!pip install -U -q tpot 
#!pip install -U -q pycaret
#!pip install -U -q tensorflow tensorboard 
#!pip install -U -q torch torchvision 
#!pip install -U -q delayed
#!pip install -U -q joblib 
#!pip install -U -q joblibspark

# COMMAND ----------

import os
import itertools
import warnings
import string
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

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
#mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 - Loading Dataset and Distinguishing Attributes

# COMMAND ----------

datapath = "./data/Titanic"
df_train = pd.read_csv(os.path.join(datapath,'kaggle_titanic_train.csv'))
df_train.head()

# COMMAND ----------

df_train.info(verbose=True, show_counts=True)

# COMMAND ----------

print(df_train.select_dtypes(include='number').columns)
print(df_train.select_dtypes(include='object').columns)
print(df_train.select_dtypes(include='category').columns)

# COMMAND ----------

print(df_train.dtypes)

# COMMAND ----------

for cat in df_train.columns:
    print(f"Number of categories in `{cat+'`:':13s} \b {df_train[cat].unique().size:3d} ")

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
    print(f"Unique values for category '{cat}':")
    for name in df_train[cat].unique()[0:20]:
        print(f"   {name}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 - Checking for missing values  
# MAGIC Remember that a careful analysis of missing values takes some time, as missing values are often not immediately identifiable. 

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

import missingno as msno
msno.bar(df_train, figsize=(12,6))

# COMMAND ----------

# MAGIC %md
# MAGIC The columns with missing values are 'Age', 'Cabin' and 'Embarked'. However, we will not drop any records but instead use different techniques to deal with these. 

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
# MAGIC ## 3.1 - Creating train & test subsets
# MAGIC Before splitting we will remove 'PassengerId' because it contains only unique values without inherent order.

# COMMAND ----------

df_train.drop(["PassengerId"], axis=1, inplace=True)
target = df_train.pop("Survived")

# COMMAND ----------

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train, target, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# COMMAND ----------

# drop old index as it is meaningless
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 - Transforming Sex (stateful/binary)
# MAGIC We start by using one hot encoding on the 'Sex' feature. This column did not have any unknown values. If it had, `handle_unknown='ignore'` makes it so that the resulting one-hot encoded columns for this feature will be all zero. 

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train['Sex'].values.reshape(-1, 1))
enc.categories_[0]

# COMMAND ----------

# MAGIC %md
# MAGIC After fitting the encoder, we create the one-hot encoded columns, drop the old 'Sex' column and concatenate the new columns to our dataframe.

# COMMAND ----------

gender_train = pd.DataFrame(enc.transform(X_train['Sex'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0], dtype=np.int8)
gender_test = pd.DataFrame(enc.transform(X_test['Sex'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0], dtype=np.int8)

X_train.drop(['Sex'], axis=1, inplace=True)
X_train = pd.concat([X_train, gender_train], axis=1)

X_test.drop(['Sex'], axis=1, inplace=True)
X_test = pd.concat([X_test, gender_test], axis=1)

X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 - Transforming Embarked (stateful)  
# MAGIC We decide to substitute the missing values using a missing value imputation technique. Simply put, we will encode the missing values as an additional category using OHE.

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train['Embarked'].values.reshape(-1, 1))

# COMMAND ----------

embarked_train = pd.DataFrame(enc.transform(X_train['Embarked'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
embarked_train.rename({np.nan: "U"}, axis='columns', inplace=True)   # Rename the column with name "nan" to avoid further problems

embarked_test = pd.DataFrame(enc.transform(X_test['Embarked'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
embarked_test.rename({np.nan: "U"}, axis='columns', inplace=True)    # Rename the column with name "nan" to avoid further problems

embarked_train[0:5]

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

# MAGIC %md
# MAGIC We now have a feature 'titles' which might be interesting for our predictions, so we will analyze it further. 

# COMMAND ----------

X_train_name.title.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC There are several unique titles. These are often not meaningful.  
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
# MAGIC Now that we have reduced the number of distinct categories, we can use one-hot encoding as before to translate the titles into binary values. 

# COMMAND ----------

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train_name['title'].values.reshape(-1, 1))

# COMMAND ----------

title_train = pd.DataFrame(enc.transform(X_train_name['title'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)
title_test = pd.DataFrame(enc.transform(X_test_name['title'].values.reshape(-1, 1)).toarray(), columns=enc.categories_[0],dtype=np.int8)

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
# MAGIC Let's examine the frequency of surnames:

# COMMAND ----------

X_train_name.surname.value_counts().plot(kind="hist", figsize=(12,4))

# COMMAND ----------

# MAGIC %md
# MAGIC Given that most of the surnames are unique, let's just use a frequency encoder.
# MAGIC As the name suggests, this type of encoder replaces the names of the groups with the group counts, potentially giving more frequent names more weight. 
# MAGIC https://contrib.scikit-learn.org/category_encoders/count.html

# COMMAND ----------

!pip install -U -q category_encoders

# COMMAND ----------

import category_encoders as ce

enc = ce.count.CountEncoder(verbose=0)
enc.fit(X_train_name.surname)

# COMMAND ----------

surnames_train = pd.DataFrame(enc.transform(X_train_name.surname), columns=enc.get_feature_names())
X_train = pd.concat([X_train, surnames_train], axis=1)

surnames_test = pd.DataFrame(enc.transform(X_test_name.surname), columns=enc.get_feature_names())
X_test = pd.concat([X_test, surnames_test], axis=1)

X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 - Transforming Ticket (stateless)

# COMMAND ----------

# MAGIC %md
# MAGIC We can indeed gather interesting information from 'Ticket' but for this demonstration we will drop it.
# MAGIC We could also try to find some relationship between 'Pclass' and 'Ticket' but their correlation is not very strong, so we decided to drop this variable as well.

# COMMAND ----------

X_train.drop(['Ticket'], axis=1, inplace=True)
X_test.drop(['Ticket'], axis=1, inplace=True)

X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 - Transforming Cabin (stateful)

# COMMAND ----------

# MAGIC %md
# MAGIC It might be tempting to discard the 'Cabin' because of the high number of missing values, as well as because of the large amount of unique values. 

# COMMAND ----------

print(f"Number of distinct 'Cabin' values: {len(X_train['Cabin'].value_counts())}")
print(f"Number of unknown values: {X_train['Cabin'].isna().sum()}")

# COMMAND ----------

# MAGIC %md The second issue, however, can be overcome by researching what the cabin numbers mean: The first letter indicates the level on which the cabin is located, whereas the number is the concrete cabin number. The level might contain significant information on the social status of the passengers, or the distance to the lifeboats.
# MAGIC 
# MAGIC ![Cabins](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Olympic_%26_Titanic_cutaway_diagram.png/800px-Olympic_%26_Titanic_cutaway_diagram.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Thus we will only use the first letter of the 'Cabin' feature. Unknown values will receive the letter 'U'.

# COMMAND ----------

def replace_cabin_with_deck(dataset):
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    dataset.drop(['Cabin'], axis=1, inplace=True)

# COMMAND ----------

replace_cabin_with_deck(X_train)
replace_cabin_with_deck(X_test)
X_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 - Transforming Age (stateful)  
# MAGIC We will use the iterative inputer wchich fills missing values using a model based on the other features. 

# COMMAND ----------

imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(X_train)

X_train = pd.DataFrame(imp_mean.transform(X_train.values), columns=X_train.columns)

X_test = pd.DataFrame(imp_mean.transform(X_test.values), columns=X_test.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.8 - Dealing with numeric fields values
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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for column in X_train.columns:
    scaler.fit(X_train[column].values.reshape(-1, 1))
    X_train[column] = scaler.transform(X_train[column].values.reshape(-1, 1))
    X_test[column] = scaler.transform(X_test[column].values.reshape(-1, 1))

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.9 - Balancing the classes  

# COMMAND ----------

def inspect_balance(target):
    class1 = len(target[y_train == 1])
    class2 = len(target[y_train == 0])
    print(f"Class 1: {class1:4d} ({class1/(class1+class2)*100:2.0f}%)")
    print(f"Class 0: {class2:4d} ({class2/(class1+class2)*100:2.0f}%)")
    print()

inspect_balance(y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC As the dataset is not too strongly unbalanced, one could experiment with running the algorithms without any balancing. However, we will oversample to improve the balance. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.9.1 - Oversampling the minority class
# MAGIC 
# MAGIC For balancing our dataset, we use SMOTE (Synthetic Minority Over-sampling TEchnique).

# COMMAND ----------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)

X_train, y_train = smote.fit_resample(X_train, y_train)

# COMMAND ----------

inspect_balance(y_train)
inspect_balance(y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4 - Metrics and Evaluation Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 - Importing Metrics 

# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV   


# If we are using a spark environment we can paralelize grid search
from joblibspark import register_spark
from sklearn.utils import parallel_backend
register_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 - Creating Cross validation folds   
# MAGIC This will make more precise the evaluation of the classifiers when doing grid search.  
# MAGIC Bear in mind that this also affects the sample used for training, making the performance slighty worse than when using the whole dataset.   
# MAGIC https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 

# COMMAND ----------

cv = model_selection.KFold(n_splits=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 - Creating functions to evaluate the models
# MAGIC 
# MAGIC The following functions can be put aside in a python file and imported, to make the code less cluttered.

# COMMAND ----------

def mean_scores_cv(clf, cv, X, y):
    print("Using f1 score")
    scores = model_selection.cross_val_score(clf, X, y, 
                                             scoring="f1", 
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
# MAGIC A linear classifier achieves a classification decision based on the value of a linear combination of the characteristics. An object's characteristics are also known as feature values and are typically presented to the machine in a vector called a *feature vector*. Such classifiers work well for practical problems such as document classification, and more generally for problems with many variables (features), reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1 - [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
# MAGIC 
# MAGIC Logistic regression is a classification algorithm based on the function which is used at the core of the method, logistic function or sigmoid function. It’s an S-shaped curve that is used to predict a binary outcome (1/0, Yes/No, True/False) given a set of independent variables.
# MAGIC 
# MAGIC + It can also be thought of as a special case of linear regression when the outcome variable is categorical, where we are using the log of odds as a dependent variable.
# MAGIC + It predicts the probability of occurrence of an event by fitting data to a logit function.
# MAGIC 
# MAGIC \\(f(x) = \frac{L}{1 + e^{-k(x-x_0)}}\\) 
# MAGIC 
# MAGIC Logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. In the binary case, a dependent variable (the output) can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.
# MAGIC 
# MAGIC 
# MAGIC ![Logistic](https://www.saedsayad.com/images/LogReg_1.png)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

clf_lr = LogisticRegression(penalty='l2',
                            dual=False, 
                            tol=0.001, 
                            C=0.10, 
                            fit_intercept=True, 
                            intercept_scaling=1, 
                            class_weight=None, 
                            random_state=0, 
                            solver='saga', 
                            max_iter=100, 
                            multi_class='ovr', 
                            verbose=0, 
                            warm_start=False, 
                            n_jobs=-1).fit(X_train, y_train)

cv_lr = mean_scores_cv(clf_lr, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_lr}\n")

roc_lr = clf_eval(clf_lr, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.1.2 - Logistic Regression with Grid Search
# MAGIC 
# MAGIC For each model, we can also run a hyperparameter search to find the best parameters.

# COMMAND ----------

estimator = LogisticRegression(dual=False, 
                               penalty='l2',
                               fit_intercept=True, 
                               intercept_scaling=1, 
                               class_weight=None, 
                               random_state=0, 
                               solver='saga', 
                               multi_class='ovr', 
                               verbose=0, 
                               warm_start=False, 
                               n_jobs=1)

Cs = np.linspace(0,1,10)
max_iter = [10, 50, 100, 200]
tol = [1e-5, 1e-4, 1e-3]
fi = [True, False]
param_grid=dict(C=Cs, max_iter=max_iter, tol=tol, fit_intercept=fi)


clf_lr2 = model_selection.GridSearchCV(param_grid=param_grid,                       ## Grid Search (more exhaustive)
#clf_lr2 = model_selection.RandomizedSearchCV(param_distributions=param_grid,       ## Randomized (faster)
                                       estimator=estimator,
                                       cv=cv,                                       ## using cross validation
                                       n_jobs=1)
clf_lr2.fit(X_train, y_train)

print(clf_lr2.best_score_)
print(clf_lr2.best_estimator_.C)
print(clf_lr2.best_estimator_.max_iter)
print(clf_lr2.best_estimator_.tol)

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
# MAGIC The precision recall curve can be used for all classifiers that expose the probabilities of prediction for the classes.  
# MAGIC For each person in the Titanic dataset the classifier algorithm has to classify, it computes a probability based on a specific function.  
# MAGIC It then classifies the person either as survived or not survived, depending on wether the score given by the function is over or under some threshold.    
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
# MAGIC The most balanced choice of threshold seems to lie around a value of 0.4. 
# MAGIC 
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
# MAGIC The ridge coefficients minimize a penalized residual sum of squares.  
# MAGIC https://stats.stackexchange.com/questions/558900/why-ridgeclassifier-can-be-significantly-faster-than-logisticregression-with-a-h

# COMMAND ----------

from sklearn.linear_model import RidgeClassifier

clf_rdg = RidgeClassifier(alpha=1.0, 
                          fit_intercept=True, 
                          normalize=False, 
                          copy_X=True, 
                          max_iter=None, 
                          tol=0.001, 
                          class_weight=None, 
                          solver='auto', 
                          random_state=0).fit(X_train, y_train)

cv_rdg = mean_scores_cv(clf_rdg, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_rdg}\n")

roc_rdg = clf_eval(clf_rdg, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.3 - [Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)
# MAGIC 
# MAGIC Perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time.

# COMMAND ----------

from sklearn.linear_model import Perceptron

clf_pcp = Perceptron(penalty=None,
                     alpha=0.001,
                     fit_intercept=True,
                     max_iter=230,
                     shuffle=True,
                     verbose=0,
                     eta0=1.0,
                     n_jobs=-1, 
                     random_state=0, 
                     class_weight=None, 
                     warm_start=False).fit(X_train, y_train)

cv_pcp = mean_scores_cv(clf_pcp, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_pcp}\n")

roc_pcp = clf_eval(clf_pcp, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.4 - [Passive Aggressive](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)  
# MAGIC 
# MAGIC https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/ 
# MAGIC 
# MAGIC Passive Aggressive Algorithms are a family of online learning algorithms (for both classification and regression) proposed by Crammer at al. The idea is very simple and their performance has been proven to be superior to many other alternative methods like Online Perceptron and MIRA.  
# MAGIC 
# MAGIC **Passive**   
# MAGIC If classification is correct, keep the model.
# MAGIC 
# MAGIC **Aggressive**   
# MAGIC If classification is incorrect, update to adjust to this misclassified example. 
# MAGIC 
# MAGIC In passive, the information hidden in the example is not enough for updating; in aggressive, the information shows that at least this time you are wrong and a better model should modify this mistake.

# COMMAND ----------

from sklearn.linear_model import PassiveAggressiveClassifier

clf_pac = PassiveAggressiveClassifier(C=0.1, 
                                      fit_intercept=True, 
                                      max_iter=100, 
                                      shuffle=True, 
                                      verbose=0, 
                                      loss='hinge', 
                                      n_jobs=-1, 
                                      random_state=0, 
                                      warm_start=False, 
                                      class_weight=None).fit(X_train, y_train)

cv_pac = mean_scores_cv(clf_pac, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_pac}\n")

roc_pac = clf_eval(clf_pac, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.5 - [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
# MAGIC 
# MAGIC The SGD classifier is a linear classifier (such as SVM, logistic regression) with SGD training (Stochastic Gradient Descent):   
# MAGIC The gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning (see the *partial_fit* method). For best results using the default learning rate schedule, the data should have zero mean and unit variance.  
# MAGIC 
# MAGIC The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.

# COMMAND ----------

from sklearn.linear_model import SGDClassifier

clf_sgdc = SGDClassifier(loss='hinge',
                         penalty='l2', 
                         alpha=0.0001,
                         l1_ratio=0.15, 
                         fit_intercept=True,
                         max_iter=200, 
                         shuffle=True,
                         verbose=0,
                         epsilon=0.01,
                         n_jobs=-1,
                         random_state=0,
                         learning_rate='optimal',
                         eta0=0.0, 
                         power_t=0.5,
                         class_weight=None,
                         warm_start=False, 
                         average=False).fit(X_train, y_train)

cv_sgdc = mean_scores_cv(clf_sgdc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_sgdc}\n")

roc_sgdc = clf_eval(clf_sgdc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1.6 - [Support Vector Machines](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/  
# MAGIC 
# MAGIC Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of binary training examples, an SVM training algorithm builds a model that assigns new observations to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the observations as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# MAGIC 
# MAGIC In addition to performing linear classification, SVMs can efficiently perform a **non-linear classification** using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
# MAGIC 
# MAGIC An SVM will find the hyperplane or a boundary between the two classes of data that maximizes the margin between the two classes. 
# MAGIC 
# MAGIC \\( B_0 + (B_1 * X_1) + (B_2 * X_2) = 0\\) where, \\(B_1\\) and \\(B_2\\) determines the slope of the line and \\(B_0\\) (intercept) found by the learning algorithm. \\(X_1\\) and\\( X_2\\) are the two input variables.
# MAGIC 
# MAGIC ![SVM](https://www.researchgate.net/publication/331308937/figure/fig1/AS:870140602249216@1584469094771/Illustration-of-support-vector-machine-SVM-to-generalize-the-optimal-separating.ppm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Choosing the best parameters using [GridSearchCV](http://scikit-learn.org/stable/modules/grid_search.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

# COMMAND ----------

from sklearn import svm

estimator = svm.SVC()

#kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernels = ['poly', 'rbf']
Cs = np.linspace(0.1,3,5)
degrees = [2,3]
gammas = np.logspace(-5, 0, 5)

#param_grid=dict(kernel=kernels, C=Cs, gamma=gammas, degree=degrees)
param_grid=dict(kernel=kernels, degree=degrees)


clf_svc = model_selection.GridSearchCV(param_grid=param_grid,                       ## Grid Search (more exhaustive)
#clf_svc = model_selection.RandomizedSearchCV(param_distributions=param_grid,       ## Randomized (faster)
                                       estimator=estimator,
                                       cv=cv,                                       ## using cross validation
                                       #n_jobs=-1
                                      )

#clf_svc.fit(X_train, y_train)
with parallel_backend('spark', n_jobs=100):     #If we have a spark backend available https://github.com/joblib/joblib-spark
    clf_svc.fit(X_train, y_train)

print(clf_svc.best_score_)
print(clf_svc.best_estimator_.kernel)
print(clf_svc.best_estimator_.C)
print(clf_svc.best_estimator_.degree)
print(clf_svc.best_estimator_.gamma)

# COMMAND ----------

cv_svc = mean_scores_cv(clf_svc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_svc}\n")

roc_svc = clf_eval(clf_svc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 - [k-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)  
# MAGIC 
# MAGIC k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# MAGIC 
# MAGIC + In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# MAGIC 
# MAGIC + In k-NN regression, the output is the average of the values of its k nearest neighbors.
# MAGIC 
# MAGIC k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.
# MAGIC 
# MAGIC It works by finding the distances between the new data point added and the points already existed in the two separate classes. Whatever class has majority, the new data point belongs to that class.
# MAGIC 
# MAGIC ![knn](https://miro.medium.com/max/1400/1*T8Pnw0kiVbrPGnqnB2I_Zw.jpeg)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=25,
                               weights='uniform', 
                               algorithm='auto', 
                               leaf_size=30, 
                               p=4, 
                               metric='minkowski', 
                               metric_params=None, 
                               n_jobs=-1).fit(X_train, y_train)

cv_knn = mean_scores_cv(clf_knn, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_knn}\n")

roc_knn = clf_eval(clf_knn, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 - [Decision Trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
# MAGIC 
# MAGIC Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# MAGIC 
# MAGIC 
# MAGIC The **Gini score** gives an idea of how good a split is by how mixed the response classes are in the groups created by the split.
# MAGIC 
# MAGIC ![dtree](https://elf11.github.io/images/decisionTree.png)

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

clf_dtc = DecisionTreeClassifier(criterion='gini', 
                                 splitter='best', 
                                 max_depth=None, 
                                 min_samples_split=3, 
                                 min_samples_leaf=1, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_features=None, 
                                 random_state=0, 
                                 max_leaf_nodes=None, 
                                 class_weight=None,).fit(X_train, y_train)

cv_dtc = mean_scores_cv(clf_dtc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_dtc}\n")

roc_dtc = clf_eval(clf_dtc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 - [Ensemble Classifiers](http://scikit-learn.org/stable/modules/ensemble.html)
# MAGIC 
# MAGIC An ensemble classifier works by combinding the predictions of several base estimators, often linear models. Such classifiers work well for practical problems such as document classification, and more generally for problems with many features, reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.
# MAGIC 
# MAGIC We have many type of ensembles: 
# MAGIC + Bagging
# MAGIC + Boosting 
# MAGIC + Voting & Stacking

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4.1 - Bagging
# MAGIC 
# MAGIC **Bootstrap aggregating**, also called **bagging**, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the **model averaging approach**. Random Forests are the most common type of bagging algorithms.

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



# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=300, 
                                criterion='gini', 
                                max_depth=None, 
                                min_samples_split=3, #2,
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='sqrt', 
                                max_leaf_nodes=None, 
                                bootstrap=True, 
                                oob_score=False, 
                                n_jobs=-1, 
                                random_state=0, 
                                verbose=0, 
                                warm_start=False, 
                                class_weight=None).fit(X_train, y_train)

cv_rf = mean_scores_cv(clf_rf, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_rf}\n")

roc_rf = clf_eval(clf_rf, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC **Assessing the importance of the features**

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
# MAGIC A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then creating an ensemble from it.

# COMMAND ----------

from sklearn.ensemble import BaggingClassifier

clf_bgc = BaggingClassifier(clf_lr).fit(X_train, y_train)   ## Using the previously fitted Logistic Regression

cv_bgc = mean_scores_cv(clf_bgc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_bgc}\n")

roc_bgc = clf_eval(clf_bgc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.1.3 - [Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
# MAGIC 
# MAGIC In extremely randomized trees, such as in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature. The best of these randomly-generated thresholds is picked as the splitting rule. 
# MAGIC This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias.

# COMMAND ----------

from sklearn.ensemble import ExtraTreesClassifier

clf_etc = ExtraTreesClassifier(n_estimators=300,
                               max_depth=None,
                               min_samples_split=3,
                               random_state=0).fit(X_train, y_train)

cv_etc = mean_scores_cv(clf_etc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_etc}\n")

roc_etc = clf_eval(clf_etc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4.2 - Boosting
# MAGIC 
# MAGIC Boosting is a machine learning ensemble meta-algorithm primarily for reducing bias, and also variance in supervised learning. It describes a family of machine learning algorithms that convert weak learners to strong ones. 
# MAGIC 
# MAGIC A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

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

clf_abc = AdaBoostClassifier(base_estimator=None,
                             n_estimators=300,
                             learning_rate=0.1,
                             algorithm='SAMME.R',
                             random_state=0).fit(X_train, y_train)

cv_abc = mean_scores_cv(clf_abc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_abc}\n")

roc_abc = clf_eval(clf_abc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.2 - [Gradient Boost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html):  
# MAGIC 
# MAGIC Gradient Boost is also an ensemble algorithm that uses boosting methods to develop an enhanced predictor.
# MAGIC 
# MAGIC + Unlike AdaBoost, which builds stumps, Gradient Boost builds trees with usually 8–32 leaves.
# MAGIC + Gradient Boost views the boosting problem as an optimization problem, where it uses a loss function and tries to minimize the error. This is why it’s called **Gradient boost, as it’s inspired by gradient descent.

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier

clf_gbc = GradientBoostingClassifier(learning_rate=0.1,
                                     n_estimators=100,
                                     subsample=1.0,
                                     criterion='friedman_mse',
                                     min_samples_split=3, 
                                     min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, 
                                     max_depth=3, 
                                     init=None, 
                                     random_state=0, 
                                     max_features=None, 
                                     verbose=0, 
                                     max_leaf_nodes=None, 
                                     warm_start=False,
                                     validation_fraction=0.1, 
                                     n_iter_no_change=None, 
                                     tol=0.0001, 
                                     ccp_alpha=0.0).fit(X_train, y_train)
                                     
cv_gbc = mean_scores_cv(clf_gbc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_gbc}\n")                                   
                                     
roc_gbc = clf_eval(clf_gbc, X_test, y_test)                                     

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

clf_xgb = xgboost.sklearn.XGBClassifier(base_score=0.5,
                                        learning_rate=0.1,
                                        n_estimators=250,
                                        max_delta_step=0,
                                        max_depth=2,
                                        min_child_weight=1,
                                        missing=1,
                                        gamma=0.1,
                                        subsample=1,
                                        colsample_bylevel=1,
                                        colsample_bytree=1,
                                        objective= 'binary:logitraw',
                                        #objective='multi:softprob',
                                        eval_metric='auc',
                                        #eval_metric='logloss',
                                        reg_alpha=0, 
                                        reg_lambda=1,
                                        nthread=-1,
                                        scale_pos_weight=1,
                                        seed=0,
                                        use_label_encoder=False,
                                        random_state=0).fit(X_train, y_train)

cv_xgb = mean_scores_cv(clf_xgb, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_xgb}\n")

roc_xgb = clf_eval(clf_xgb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.4 - Light GBM: 
# MAGIC 
# MAGIC It is another type of boosting algorithm that has shown to be faster and sometimes more accurate than XGBoost.
# MAGIC 
# MAGIC + It uses a unique technique called Gradient-based One-side sampling (GOSS) to filter out the data instances to find a split value.

# COMMAND ----------

import lightgbm as lgb

params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'num_leaves': 64, 
          'learning_rate': 0.05, 
          'max_bin': 512, 
          'subsample_for_bin': 200,
          'subsample': 1, 
          'subsample_freq': 1, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 5, 
          'reg_lambda': 10,
          'min_split_gain': 0.15, 
          'min_child_weight': 1, 
          'min_child_samples': 5, 
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

clf_lgb = lgb.LGBMClassifier(boosting_type= 'gbdt',
                             objective = 'binary',
                             n_jobs = -1,
                             max_depth = params['max_depth'],
                             max_bin = params['max_bin'],
                             subsample_for_bin = params['subsample_for_bin'],
                             subsample = params['subsample'],
                             subsample_freq = params['subsample_freq'],
                             min_split_gain = params['min_split_gain'], 
                             min_child_weight = params['min_child_weight'], 
                             min_child_samples = params['min_child_samples'], 
                             scale_pos_weight = params['scale_pos_weight'])

clf_lgb.fit(X_train, y_train)


cv_lgb = mean_scores_cv(clf_lgb, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_lgb}\n")

roc_lgb = clf_eval(clf_lgb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.2.5 - [Catboost](https://catboost.ai/)  

# COMMAND ----------

from catboost import CatBoostClassifier
#https://catboost.ai/en/docs/concepts/parameter-tuning

tmp_path = '/tmp'

clf_ctb = CatBoostClassifier(iterations=5,
                             learning_rate=0.01,
                             depth=7,
                             l2_leaf_reg=6,
                             train_dir=tmp_path)
clf_ctb.fit(X_train, y_train, verbose=False)

cv_ctb = mean_scores_cv(clf_ctb, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_ctb}\n")

roc_ctb = clf_eval(clf_ctb, X_test, y_test)

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

clf_rgf = RGFClassifier(max_leaf=100,
                        algorithm="RGF_Sib",
                        test_interval=60,
                        verbose=False,).fit(X_train, y_train)

cv_rgf = mean_scores_cv(clf_rgf, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_rgf}\n")

roc_rgf = clf_eval(clf_rgf, X_test, y_test)

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

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ensemble = VotingClassifier(estimators=[('clf_sgdc', clf_sgdc),
                                            ('clf_lgr', clf_lr),
                                            ('clf_rdg', clf_rgf),
                                            ('clf_bgc', clf_bgc),
                                            ('clf_etc', clf_etc),
                                            ('clf_abc', clf_abc),
                                            ('clf_pct', clf_pcp),
                                            ('clf_xgb', clf_xgb),
                                            ('clf_rf', clf_rf),
                                            ('clf_knn', clf_knn),
                                            ('clf_rgf', clf_rgf),
                                            ('clf_ctb', clf_ctb),
                                            #('clf_tpot', clf_tpot),
                                            ],
                                voting='hard',
                                weights=[1,1,1,1,1,1,1,1,1,1,1,1]).fit(X_train, y_train)

cv_ens = mean_scores_cv(ensemble, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_ens}\n")
    
roc_ens = clf_eval(ensemble, X_test, y_test)

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

clf_bnb = BernoulliNB(alpha=0.20, 
                     binarize=0.0, 
                     fit_prior=True, 
                     class_prior=None).fit(X_train, y_train)

cv_bnb = mean_scores_cv(clf_bnb, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_bnb}\n")

roc_bnb = clf_eval(clf_bnb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 - Gaussian Processes
# MAGIC 
# MAGIC In the simple linear regression setting, we have a dependent variable y that we assume can be modeled as a function of an independent variable x, i.e. y=f(x)+ϵ (where ϵ is the irreducible error) but we assume further that the function f defines a linear relationship and so we are trying to find the parameters $θ_0$ and $θ_1$ which define the intercept and slope of the line respectively, i.e. $y=θ_0+θ_1x+ϵ$.
# MAGIC + Bayesian linear regression provides a probabilistic approach to this by finding a distribution over the parameters that gets updated whenever new data points are observed. 
# MAGIC + The GP approach, in contrast, is a non-parametric approach, in that it finds a distribution over the possible functions $f(x)$ that are consistent with the observed data. As with all Bayesian methods it begins with a prior distribution and updates this as data points are observed, producing the posterior distribution over functions.

# COMMAND ----------

from sklearn.gaussian_process import GaussianProcessClassifier

clf_gpc = GaussianProcessClassifier(kernel=None, 
                                    optimizer='fmin_l_bfgs_b', 
                                    n_restarts_optimizer=0, 
                                    max_iter_predict=100, 
                                    warm_start=False, 
                                    copy_X_train=True, 
                                    random_state=0, 
                                    multi_class='one_vs_rest', 
                                    n_jobs=-1).fit(X_train, y_train)

cv_gpc = mean_scores_cv(clf_gpc, cv, X_train, y_train)
print(f"This is the cross validated score of the model: {cv_gpc}\n")

roc_gpc = clf_eval(clf_gpc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.7 - Neural Networks
# MAGIC 
# MAGIC Neural Networks take inspiration from the learning process occurring in human brains. They consists of an artificial network of functions, called parameters, which allows the computer to learn, and to fine tune itself, by analyzing new data. Each "neuron" learns the parameters of a function which produces an output, after receiving one or multiple inputs. Those outputs are then passed to the next layer of neurons, which use them as inputs of their own function, and produce further outputs. Those outputs are then passed on to the next layer of neurons, and so it continues until every layer of neurons have been considered, and the terminal neurons have received their input. Those terminal neurons then output the final result for the model.  

# COMMAND ----------

import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

batch_size = int(len(X_train)/25)
num_classes = 2
epochs = 30
np.random.seed(0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(input_dim=X_train.shape[1], units=20, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=40, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[callback],
                    verbose=0,
                    shuffle=False,
                    validation_split=0.2,
                    #validation_data=(X_test, y_test)
                   )

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = (model.predict(X_test) > 0.5).astype("int32")


clf_matrix = confusion_matrix(y_test, y_pred)
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=['Perished', 'Survived']))
print('ROC Score: {}'.format(roc_auc_score(y_test, y_pred)))
print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))
print('Average Precision Score: {}'.format(average_precision_score(y_test, y_pred)))
print('f1 Score: {}'.format(f1_score(y_test, y_pred)))
plot_confusion_matrix(clf_matrix, classes=['Perished', 'Survived'])
roc_keras = roc_auc_score(y_test, y_pred)

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
# MAGIC 
# MAGIC **Automated Machine Learning** is a technology to automate machine learning tasks, help data scientists focus on higher value duties, and improve the accuracy of ML models. 
# MAGIC Basically, AutoML automatically selects the model algorithm, hyperparameter optimization, models by iterations, and model evaluation.   
# MAGIC This technology doesn’t substitute data scientists but frees them from repetitive tasks.
# MAGIC 
# MAGIC A **genetic algorithm** is an adaptive heuristic search algorithm used to solve optimization problems in machine learning. It helps solve complex problems that would otherwise take a long time to solve.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.8.1 - [TPOT](https://github.com/rhiever/tpot)  
# MAGIC 
# MAGIC TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

# COMMAND ----------

from tpot import TPOTClassifier

mlflow.autolog(disable=True)

clf_tpot = TPOTClassifier(verbosity=1, 
                          max_time_mins=2, 
                          max_eval_time_mins=10, 
                          population_size=100,
                          generations=100,
                          n_jobs=-1)

clf_tpot.fit(X_train, y_train)

roc_tpot = clf_eval(clf_tpot, X_test, y_test)

mlflow.autolog(disable=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.8.2 - [PyCaret](https://pycaret.gitbook.io/docs/)  
# MAGIC 
# MAGIC https://pycaret.gitbook.io/docs/get-started/quickstart  
# MAGIC PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows.  
# MAGIC It is an end-to-end machine learning and model management tool that exponentially speeds up the experiment cycle and makes you more productive.

# COMMAND ----------

# from pycaret.classification import *
# s = setup(pd.concat([X_train, y_train], axis=1), target="Survived")
# best = compare_models()

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
               'Catboost': roc_ctb,
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
                      'clf_bgc':(abs(clf_lr.coef_[0])/sum(abs(clf_lr.coef_[0]))),   # Adapting to linear regression classifiers in our bagging strategy
                      'clf_gbc':clf_gbc.feature_importances_,
                      'clf_lgb':clf_lgb.feature_importances_ / sum(clf_lgb.feature_importances_),
                      'clf_ctb': clf_ctb.feature_importances_  / sum(clf_ctb.feature_importances_),
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

title = f'''Learning Curves (Random Forest, 
Estimators:{clf_rfo.best_estimator_.n_estimators},
Max Depth:{clf_rfo.best_estimator_.max_depth},
Min Samples Split:{clf_rfo.best_estimator_.min_samples_split},
Min Samples Leaf:{clf_rfo.best_estimator_.min_samples_leaf},
Criterion:{clf_rfo.best_estimator_.criterion})'''
graph = plot_learning_curve(clf_rfo, title, X_train, y_train, cv=cv)
graph.show()
