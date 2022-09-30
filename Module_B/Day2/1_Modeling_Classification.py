# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction to Data Science
# MAGIC 
# MAGIC ## Predictive Analysis - numerical and categorical data
# MAGIC 
# MAGIC ### The sinking of Titanic  
# MAGIC Based on [this](https://www.kaggle.com/c/titanic-gettingStarted) Kaggle Competition. Inspired by a handful of solutions, like [this one](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8)

# COMMAND ----------

#!pip install -U -q forestci rgf_python scikit-learn joblib tpot imbalanced-learn tensorflow tensorboard xgboost lightgbm torch torchvision delayed

# COMMAND ----------

import os
import sys
import time
import pickle
import itertools
import pandas as pd
import numpy as np
import pylab
import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

from IPython.display import display, Image
from IPython.core.interactiveshell import InteractiveShell

%matplotlib inline
#%matplotlib notebook
#matplotlib.rcdefaults()
#matplotlib.verbose.set_level('silent')

# COMMAND ----------

## Optional -- testing GPU support to tensorflow
import tensorflow as tf
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Mighty Titanic !
# MAGIC 
# MAGIC ![Titanic](../Images/titanic.jpeg)

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
# MAGIC ### Importing the datasets

# COMMAND ----------

datapath = "../Data/Kaggle/Titanic"
outputs = "../Data/Kaggle/Titanic"

# COMMAND ----------

df_train = pd.read_csv(os.path.join(datapath,'kaggle_titanic_train.csv'))
df_test = pd.read_csv(os.path.join(datapath,'kaggle_titanic_test.csv'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Data

# COMMAND ----------

df_train.head()

# COMMAND ----------

df_train.info()

# COMMAND ----------

df_train.dtypes

# COMMAND ----------

df_train.dtypes[df_train.dtypes.map(lambda x: x=='int64')]

# COMMAND ----------

df_train.dtypes[df_train.dtypes.map(lambda x: x=='float64')]

# COMMAND ----------

df_train.dtypes[df_train.dtypes.map(lambda x: x=='object')]

# COMMAND ----------

df_train.columns

# COMMAND ----------

for cat in df_train.columns:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, df_train[cat].unique().size))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Describing the numeric values

# COMMAND ----------

df_train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examining the categorical values

# COMMAND ----------

# There are many values for name and ticket

for cat in ['Sex', 'Survived', 'Pclass', 'SibSp', 'Embarked', 'Cabin']:
    print("Unique values for category '{0}': \b {1} ".format(cat, df_train[cat].unique()))

# COMMAND ----------

print(df_train.Survived.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Survived by sex:

# COMMAND ----------

df_train.groupby('Sex').Survived.value_counts()

# COMMAND ----------

df_by_sex = df_train.groupby('Sex')
df_by_sex.describe()

# COMMAND ----------

# Split the survived passengers to male and female
males = df_train[df_train['Sex'] == 'male']
survived_males = df_train[(df_train['Sex']=='male')&(df_train['Survived']==1)]

females = df_train[df_train['Sex'] == 'female']
survived_females = df_train[(df_train['Sex']=='female')&(df_train['Survived']==1)]

# COMMAND ----------

print(females["Survived"].value_counts())

sns.kdeplot(females['Age'], label = 'all females', shade = False, alpha = 0.8)
sns.kdeplot(survived_females['Age'], label = 'survived females', shade = False, alpha = 0.8)

# label the plot
plt.xlabel('Age', size = 20)
plt.ylabel('Density', size = 20)
plt.title('Density Plot of Female Titanic Passengers by Age', size = 20)

# COMMAND ----------

print(males["Survived"].value_counts())

sns.kdeplot(males['Age'], label = 'all males', shade = False, alpha = 0.8)
sns.kdeplot(survived_males['Age'], label = 'survived males', shade = False, alpha = 0.8)

# label the plot
plt.xlabel('Age', size = 20); plt.ylabel('Density', size = 20)
plt.title('Density Plot of Male Titanic Passengers by Age', size = 20)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Survived by Pclass

# COMMAND ----------

df_train.groupby('Pclass').Survived.value_counts()

# COMMAND ----------

df_train.groupby(['Pclass']).Survived.value_counts()

# COMMAND ----------

df_by_class = df_train.groupby('Pclass')
df_by_class.describe()

# COMMAND ----------

sns.barplot(x='Pclass', y='Survived', data=df_train)

# COMMAND ----------

grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=4.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# COMMAND ----------

# MAGIC %md
# MAGIC #### Survived by Age

# COMMAND ----------

#df_train['Age'].hist()
df_train['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5, figsize=(10,6))

# COMMAND ----------

ser, bins = pd.qcut(df_train.Age.dropna(), 5, retbins=True, labels=False)
print(bins)
df_train.groupby(ser).Survived.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sex and Pclass

# COMMAND ----------

df_train.groupby(['Pclass']).Sex.value_counts()

# COMMAND ----------

df_train.groupby(['Pclass','Sex']).Survived.value_counts()

# COMMAND ----------

id = pd.crosstab([df_train.Pclass, df_train.Sex], df_train.Survived.astype(float))
id.columns = (['No', 'Yes'])
id.columns.name = "Survived"
id.div(id.sum(1).astype(float), 0)

# COMMAND ----------

# Set-up 3x2 grid for plotting.

fig, sub = plt.subplots(1,3, figsize=(14,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

titles = [" First Class "," Second Class "," Third Class "]

for i, title, ax in zip([1,2,3], titles, sub.flatten()):
    male_data = males[males['Pclass']==i]
    female_data = females[females['Pclass']==i]
    
    plt_title = title + 'Passengers'
    sns.kdeplot(male_data['Age'],
               label = 'male', shade = False, alpha = 0.8,ax=ax);
    sns.kdeplot(female_data['Age'],
               label = 'female', shade = False, alpha = 0.8, ax=ax)
    
    ax.set_xlabel('Age', size = 14)
    ax.set_ylabel('Density', size = 14)
    ax.set_title(plt_title, size = 16)
    ax.set(xlim=(0))
    
plt.show()

# COMMAND ----------

# Set-up 3x2 grid for plotting.

fig, sub = plt.subplots(1,3, figsize=(14,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

titles = [" 1st Class "," 2nd Class "," 3rd Class "]

for i, title, ax in zip([1,2,3], titles, sub.flatten()):
    male_data = survived_males[survived_males['Pclass']==i]
    female_data = survived_females[survived_females['Pclass']==i]
    
    plt_title = 'Surv.'+ title + 'Passengers'
    sns.kdeplot(male_data['Age'], label='male', shade=False, alpha=0.8,ax=ax);
    ax.set_xlabel('Age', size=14)
    
    sns.kdeplot(female_data['Age'], label='female', shade=False, alpha=0.8, ax=ax)
    ax.set_xlabel('Age', size=14)
    
    ax.set_ylabel('Density', size=14)
    ax.set_title(plt_title, size=16)
    ax.set(xlim=(0))
    
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dealing with missing values

# COMMAND ----------

total = df_train.isnull().sum().sort_values(ascending=False)
percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Encoding the "Sex" field

# COMMAND ----------

df_train['Gender'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_test['Gender'] = df_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filling the null values for Age

# COMMAND ----------

print(len(df_train[df_train['Age'].isnull()]))
print(len(df_test[df_test['Age'].isnull()]))

# COMMAND ----------

df_train['AgeFill'] = df_train['Age']
df_test['AgeFill'] = df_test['Age']

# COMMAND ----------

df_train[df_train['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(4)

# COMMAND ----------

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df_train[(df_train['Gender'] == i) & (df_train['Pclass'] == j+1)]['Age'].dropna().median()

median_ages

# COMMAND ----------

for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.Gender == i) & (df_train.Pclass == j+1),'AgeFill'] = median_ages[i,j]
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1),'AgeFill'] = median_ages[i,j]

# COMMAND ----------

df_train[df_train['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filling the null values for "Embarked"

# COMMAND ----------

df_train[df_train['Embarked'].isnull()]

# COMMAND ----------

df_train[59:64]

# COMMAND ----------

df_train[826:832]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Associating the missing values to the most likely class

# COMMAND ----------

df_train['Embarked'] = df_train['Embarked'].map({np.nan:1,'C':1, 'Q':2,'S':3} ).astype(int)
df_test['Embarked'] = df_test['Embarked'].map({np.nan:1,'C':1, 'Q':2,'S':3} ).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC #### One-Hot-Enconding the field "Embarked"

# COMMAND ----------

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
lb = LabelBinarizer()
mlb = MultiLabelBinarizer()

# COMMAND ----------

df_train.groupby(['Embarked']).Survived.value_counts()

# COMMAND ----------

df_train.groupby(['Embarked']).Fare.mean()

# COMMAND ----------

embarked = pd.DataFrame(lb.fit_transform(df_train['Embarked'].values), columns=['Emb1','Emb2','Emb3'])
df_train = pd.concat([df_train, embarked], axis=1)

# COMMAND ----------

embarked = pd.DataFrame(lb.fit_transform(df_test['Embarked'].values), columns=['Emb1','Emb2','Emb3'])
df_test = pd.concat([df_test, embarked], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### One-Hot-Enconding the field "Cabin"

# COMMAND ----------

df_train['Cabin'].value_counts()

# COMMAND ----------

CabinTrans = pd.DataFrame(mlb.fit_transform([{str(val)} for val in df_train['Cabin'].values]))

# COMMAND ----------

CabinTrans.head()

# COMMAND ----------

#df_train = pd.concat([df_train, CabinTrans], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of using the Cabin feature as it is - using traditional hot-encoding, we can try using only the letters.  
# MAGIC After doing some research, we could know that a cabin number looks like ‘C123’ and the letter refers to the deck.  
# MAGIC + We’re going to extract these and create a new feature, that contains a persons deck.  
# MAGIC + And then we will convert the feature into a numeric variable.

# COMMAND ----------

import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [df_train, df_test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# COMMAND ----------

df_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating a feature for number of relatives

# COMMAND ----------

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass

# COMMAND ----------

axes = sns.catplot(x='FamilySize', y='Survived', kind='point' , data=df_train, aspect = 2.5, )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extract the Titles from "Name", so that we can build a new feature out of that. (TBD)

# COMMAND ----------

#data = [df_train, df_test]
#titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

#for dataset in data:
#    # extract titles
#    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#    # replace titles with a more common title or as Rare
#    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
#                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
#    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
#    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#    # convert titles into numbers
#    dataset['Title'] = dataset['Title'].map(titles)
#    # filling NaN with 0, to get safe
#    dataset['Title'] = dataset['Title'].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fare per Person (TBD)

# COMMAND ----------

#for dataset in data:
#    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['FamilySize']+1)
#    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

#df_train.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring "Ticket"

# COMMAND ----------

df_train['Ticket'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories. So we will drop it from the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discarding unused columns for predictions

# COMMAND ----------

df_train2 = df_train.drop(['Age','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_train2 = df_train2.dropna()

df_test2 = df_test.drop(['Age','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_test2 = df_test2.dropna()

# COMMAND ----------

df_train2.head()

# COMMAND ----------

df_test2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Running some correlation hypothesis:

# COMMAND ----------

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
## https://github.com/statsmodels/statsmodels/issues/3931 waiting for fixes

y = df_train2.Survived
X = df_train2.Gender
model = sm.Logit(y, X)
results = model.fit()
print(results.summary())

# COMMAND ----------

y = df_train2.Survived
X = df_train2[['Gender','Pclass','AgeFill']]
model = sm.Logit(y, X)
results = model.fit()
print(results.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing Data for predictions

# COMMAND ----------

# MAGIC %md
# MAGIC #### Renaming target class (For the TPOT genetic algorithm) 

# COMMAND ----------

df_train2.rename(columns={'Survived': 'class'}, inplace=True)
df_train2.head(3)

# COMMAND ----------

df_test2.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Numpy vectors and adjusting features scales:   
# MAGIC 
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html  

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_train2.iloc[:,2:] = scaler.fit_transform(df_train2.iloc[:,2:])  #Excludes PassengerId and class
df_test2.iloc[:,1:] = scaler.fit_transform(df_test2.iloc[:,1:])  #Excludes PassengerId

# COMMAND ----------

X = df_train2.iloc[:,2:].values
y = df_train2.iloc[:,1].values
X_val = df_test2.iloc[:,1:].values

# COMMAND ----------

print(X.shape)
print(y.shape)
print(X_val.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating cross-validation subsets:  
# MAGIC 
# MAGIC http://scikit-learn.org/stable/modules/cross_validation.html  
# MAGIC http://www.analyticsvidhya.com/blog/2015/05/k-fold-cross-validation-simple/  
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html  
# MAGIC http://stackoverflow.com/questions/25375203/identical-learning-curves-on-subsequent-runs-using-shufflesplit  
# MAGIC http://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn  

# COMMAND ----------

from sklearn import model_selection

## Creating validation set with cross validation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

## Using stratified k-folds
#skf = model_selection.StratifiedKFold(n_splits=1)
#skf.get_n_splits(X, y)

cv = model_selection.StratifiedKFold(n_splits=5)


def mean_scores_cv(clf, cv, X, y):
    scores = model_selection.cross_val_score(clf, X, y, 
                                              scoring=None, 
                                              cv=cv, 
                                              n_jobs=1,
                                              verbose=0,
                                              fit_params=None,
                                              pre_dispatch='2*n_jobs')
    return scores.mean()

# COMMAND ----------

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Balancing the classes

# COMMAND ----------

print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Oversampling the minority class  
# MAGIC http://contrib.scikit-learn.org/imbalanced-learn/  

# COMMAND ----------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)

X_train, y_train = smote.fit_resample(X_train, y_train)

# COMMAND ----------

print(X_train.shape)
print(y_train.shape)
print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing modules for evaluation of the models
# MAGIC 
# MAGIC http://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat   
# MAGIC http://scikit-learn.org/stable/modules/cross_validation.html  

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating functions to help evaluate the models

# COMMAND ----------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,8))
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
# MAGIC ## Testing classifiers of different families:  
# MAGIC 
# MAGIC ### Linear Classifiers
# MAGIC 
# MAGIC A linear classifier achieves this by making a classification decision based on the value of a linear combination of the characteristics. An object's characteristics are also known as feature values and are typically presented to the machine in a vector called a feature vector. Such classifiers work well for practical problems such as document classification, and more generally for problems with many variables (features), reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Support Vector Machines](https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/)  
# MAGIC 
# MAGIC Support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# MAGIC 
# MAGIC In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
# MAGIC 
# MAGIC An SVM will find a hyperplane or a boundary between the two classes of data that maximizes. There are other planes as well which can separate the two classes, but only the SVM hyperplane can maximize the margin between the classes.
# MAGIC 
# MAGIC B0 + (B1 * X1) + (B2 * X2) = 0 where, B1 and B2 determines the slope of the line and B0 (intercept) found by the learning algorithm. X1 and X2 are the two input variables.
# MAGIC 
# MAGIC ![SVM](../Images/SVM.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Choosing the best parameters using [GridSearchCV](http://scikit-learn.org/stable/modules/grid_search.html) or [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)  
# MAGIC   

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC #http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# MAGIC from sklearn import svm
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
# MAGIC ## Grid Search (more exhaustive)
# MAGIC #clf_svc = model_selection.GridSearchCV(estimator=estimator,
# MAGIC #                                       cv=cv,
# MAGIC #                                       param_grid=param_grid, 
# MAGIC #                                       n_jobs=-1).fit(X_train, y_train)
# MAGIC 
# MAGIC ## Randomized (faster)
# MAGIC clf_svc = model_selection.RandomizedSearchCV(estimator=estimator,
# MAGIC                                              cv=cv,
# MAGIC                                              param_distributions=param_grid, 
# MAGIC                                              n_jobs=-1).fit(X_train, y_train)
# MAGIC 
# MAGIC 
# MAGIC with open(os.path.join(outputs,'best_parameters_svm.pickle'), 'wb') as f:
# MAGIC     pickle.dump(clf_svc,f)
# MAGIC 
# MAGIC with open(os.path.join(outputs,'best_parameters_svm.pickle'), 'rb') as f:
# MAGIC     clf_svc = pickle.load(f)
# MAGIC 
# MAGIC print(clf_svc.best_score_)
# MAGIC print(clf_svc.best_estimator_.kernel)
# MAGIC print(clf_svc.best_estimator_.C)
# MAGIC print(clf_svc.best_estimator_.degree)
# MAGIC print(clf_svc.best_estimator_.gamma)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Now, let's run with the best hiperparameters

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC clf_svc2 = svm.SVC(kernel=clf_svc.best_estimator_.kernel,
# MAGIC                    C=clf_svc.best_estimator_.C,
# MAGIC                    degree=clf_svc.best_estimator_.degree, 
# MAGIC                    gamma=clf_svc.best_estimator_.gamma, 
# MAGIC                    coef0=0.0, 
# MAGIC                    shrinking=True, 
# MAGIC                    probability=False, 
# MAGIC                    tol=0.001, 
# MAGIC                    cache_size=200, 
# MAGIC                    class_weight=None, 
# MAGIC                    verbose=False, 
# MAGIC                    max_iter=-1, 
# MAGIC                    random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_svc2 = clf_eval(clf_svc2, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### For this first classifier, we'll also display the [learning curve](http://scikit-learn.org/stable/modules/generated/sklearn.learning_curve.learning_curve.html)  

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC def plot_learning_curve(estimator, 
# MAGIC                         title, 
# MAGIC                         X, 
# MAGIC                         y, 
# MAGIC                         ylim=None, 
# MAGIC                         cv=None,
# MAGIC                         n_jobs=-1, 
# MAGIC                         train_sizes=np.linspace(.1, 1.0, 5)):
# MAGIC     
# MAGIC     plt.figure(figsize=(16,8))
# MAGIC     plt.title(title)
# MAGIC     if ylim is not None:
# MAGIC         plt.ylim(*ylim)
# MAGIC     plt.xlabel("Training examples")
# MAGIC     plt.ylabel("Score")
# MAGIC     train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, 
# MAGIC                                                             X, y, cv=cv, 
# MAGIC                                                             n_jobs=n_jobs, 
# MAGIC                                                             train_sizes=train_sizes)
# MAGIC     train_scores_mean = np.mean(train_scores, axis=1)
# MAGIC     train_scores_std = np.std(train_scores, axis=1)
# MAGIC     test_scores_mean = np.mean(test_scores, axis=1)
# MAGIC     test_scores_std = np.std(test_scores, axis=1)
# MAGIC     
# MAGIC     plt.grid()
# MAGIC     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
# MAGIC                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
# MAGIC     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
# MAGIC                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
# MAGIC     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# MAGIC     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# MAGIC 
# MAGIC     plt.legend(loc="best")
# MAGIC     return plt

# COMMAND ----------

title = 'Learning Curves (SVM, kernel:{1}{0} , $\gamma={2:.6f}$)'.format(clf_svc.best_estimator_.degree,
                                                                         clf_svc.best_estimator_.kernel,
                                                                         clf_svc.best_estimator_.gamma)
graph = plot_learning_curve(clf_svc2, title, X_train, y_train, cv=cv)
#matplotlib.rcdefaults()
#matplotlib.verbose.set_level('silent')
graph.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
# MAGIC 
# MAGIC Logistic regression is a classification algorithm based on the function which is used at the core of the method, logistic function or sigmoid function. It’s an S-shaped curve that is used to predict a binary outcome (1/0, Yes/No, True/False) given a set of independent variables.
# MAGIC 
# MAGIC + It can also be thought of as a special case of linear regression when the outcome variable is categorical, where we are using the log of odds as a dependent variable.
# MAGIC + It predicts the probability of occurrence of an event by fitting data to a logit function.
# MAGIC 
# MAGIC p(X) = e^(b0 + b1*X) / (1 + e^(b0 + b1*X))
# MAGIC 
# MAGIC Logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. In the binary case, a dependent variable (the output) can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in multinomial logistic regression, or, if the multiple categories are ordered, in ordinal logistic regression.
# MAGIC 
# MAGIC 
# MAGIC ![Logistic](../Images/logistic.jpeg)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.linear_model import LogisticRegression
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
# MAGIC #### Assessing the importance of the features

# COMMAND ----------

coefs = pd.Series(clf_lr.coef_[0], index=df_train2.iloc[:,2:].columns)
coefs = coefs.sort_values()
coefs.plot(kind="bar", figsize=(10,6))
print(coefs.sort_values(ascending = True))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Precision Recall Curve
# MAGIC 
# MAGIC For each person the classifier algorithm has to classify, it computes a probability based on a function and it classifies the person as survived (when the score is bigger the than threshold) or as not survived (when the score is smaller than the threshold). That’s why the threshold plays an important part.
# MAGIC 
# MAGIC We will plot the precision and recall with the threshold using matplotlib:

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
# MAGIC Another way is to plot the precision and recall against each other:

# COMMAND ----------

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Ridge Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)  
# MAGIC 
# MAGIC Ridge Classifiers addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.linear_model import RidgeClassifier
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
# MAGIC #### [Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)
# MAGIC 
# MAGIC Perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.linear_model import Perceptron
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
# MAGIC #### [Passive Aggressive](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)  
# MAGIC 
# MAGIC https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/ 
# MAGIC 
# MAGIC Passive Aggressive Algorithms are a family of online learning algorithms (for both classification and regression) proposed by Crammer at al. The idea is very simple and their performance has been proofed to be superior to many other alternative methods like Online Perceptron and MIRA (see the original paper in the reference section). Passive: if correct classification, keep the model; Aggressive: if incorrect classification, update to adjust to this misclassified example. In my mind, in passive, the information hidden in the example is not enough for updating; in aggressive, the information shows that at lest this time you are wrong, a better model should modify this mistake.   

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.linear_model import PassiveAggressiveClassifier
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
# MAGIC #### [SGDC Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
# MAGIC 
# MAGIC The SGDC classifier is a linear classifier (SVM, logistic regression) with SGD training. Implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.  
# MAGIC 
# MAGIC The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection.  

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.linear_model import SGDClassifier
# MAGIC 
# MAGIC clf_sgdc = SGDClassifier(loss='hinge',
# MAGIC                          penalty='l2', 
# MAGIC                          alpha=0.0001,
# MAGIC                          l1_ratio=0.15, 
# MAGIC                          fit_intercept=True,
# MAGIC                          max_iter=200, 
# MAGIC                          shuffle=True,
# MAGIC                          verbose=0,
# MAGIC                          epsilon=0.1,
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
# MAGIC ### [k-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)  
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
# MAGIC ![knn](../Images/knn.jpeg)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.neighbors import KNeighborsClassifier
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
# MAGIC ### [Decision Trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
# MAGIC 
# MAGIC Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.
# MAGIC 
# MAGIC Decision tree algorithms are referred to as CART or Classification and Regression Trees. It is a flowchart like a tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
# MAGIC 
# MAGIC + A Gini score gives an idea of how good a split is by how mixed the response classes are in the groups created by the split.
# MAGIC 
# MAGIC ![dtree](../Images/dtrees.jpeg)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.tree import DecisionTreeClassifier
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
# MAGIC ### Ensemble Classifiers
# MAGIC 
# MAGIC A linear classifier achieves this by making a classification decision based on the value of a linear combination of the characteristics. An object's characteristics are also known as feature values and are typically presented to the machine in a vector called a feature vector. Such classifiers work well for practical problems such as document classification, and more generally for problems with many variables (features), reaching accuracy levels comparable to non-linear classifiers while taking less time to train and use.
# MAGIC 
# MAGIC We have many type of ensembles: 
# MAGIC + Bagging
# MAGIC + Boosting 
# MAGIC + Voting 
# MAGIC + Stacking  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bagging
# MAGIC 
# MAGIC http://scikit-learn.org/stable/modules/ensemble.html
# MAGIC 
# MAGIC Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach. Random Forests are the most common type of bagging algorithms.

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Random Forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
# MAGIC 
# MAGIC http://blog.yhathq.com/posts/random-forests-in-python.html  
# MAGIC http://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/  
# MAGIC 
# MAGIC Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# MAGIC 
# MAGIC + Random forests involve creating multiple decision trees using bootstrapped datasets of the original data and randomly selecting a subset of variables at each step of the decision tree.     
# MAGIC + The model then selects the mode of all of the predictions of each decision tree (bagging).

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import RandomForestClassifier
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

features = df_train2.iloc[:,2:].columns
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
# MAGIC #### [Plotting the confidence intervals](https://github.com/scikit-learn-contrib/forest-confidence-interval)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import BaggingClassifier
# MAGIC 
# MAGIC clf_bgc = BaggingClassifier().fit(X_train, y_train)
# MAGIC 
# MAGIC roc_bgc = clf_eval(clf_bgc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Extra Trees Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)  

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import ExtraTreesClassifier
# MAGIC 
# MAGIC clf_etc = ExtraTreesClassifier(n_estimators=300,
# MAGIC                                max_depth=None,
# MAGIC                                min_samples_split=3,
# MAGIC                                random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_etc = clf_eval(clf_etc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boosting
# MAGIC 
# MAGIC Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones. Boosting is based on the question posed by Kearns and Valiant (1988, 1989): Can a set of weak learners create a single strong learner? A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.
# MAGIC 
# MAGIC 
# MAGIC #### [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
# MAGIC 
# MAGIC Adaptive boost is also an ensemble algorithm that leverages bagging and boosting methods to develop an enhanced predictor.
# MAGIC 
# MAGIC + AdaBoost creates a forest of stumps rather than trees. A stump is a tree that is made of only one node and two leaves.
# MAGIC + AdaBoost takes a more iterative approach in the sense that it seeks to iteratively improve from the mistakes that the previous stump(s) made.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import AdaBoostClassifier
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
# MAGIC #### [Gradient Boost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html):  
# MAGIC 
# MAGIC Gradient Boost is also an ensemble algorithm that uses boosting methods to develop an enhanced predictor.
# MAGIC 
# MAGIC + Unlike AdaBoost which builds stumps, Gradient Boost builds trees with usually 8–32 leaves.
# MAGIC + Gradient Boost views the boosting problem as an optimization problem, where it uses a loss function and tries to minimize the error. This is why it’s called Gradient boost, as it’s inspired by gradient descent.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import GradientBoostingClassifier
# MAGIC 
# MAGIC clf_gbc = GradientBoostingClassifier(loss='log_loss',
# MAGIC                                      learning_rate=0.1,
# MAGIC                                      n_estimators=200,
# MAGIC                                      subsample=1.0, 
# MAGIC                                      min_samples_split=3, 
# MAGIC                                      min_samples_leaf=1, 
# MAGIC                                      min_weight_fraction_leaf=0.0, 
# MAGIC                                      max_depth=3, 
# MAGIC                                      init=None, 
# MAGIC                                      random_state=0, 
# MAGIC                                      max_features=None, 
# MAGIC                                      verbose=0, 
# MAGIC                                      max_leaf_nodes=None, 
# MAGIC                                      warm_start=False,).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_gbc = clf_eval(clf_gbc, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### XGBoost:
# MAGIC 
# MAGIC https://github.com/dmlc/xgboost/tree/master/python-package  
# MAGIC https://xgboost.readthedocs.io/en/latest/build.html#building-on-ubuntu-debian  
# MAGIC http://xgboost.readthedocs.io/en/latest/build.html#python-package-installation  
# MAGIC http://xgboost.readthedocs.io/en/latest/parameter.html  
# MAGIC https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  
# MAGIC https://www.kaggle.com/cbrogan/titanic/xgboost-example-python/run/1620  
# MAGIC http://xgboost.readthedocs.io/en/latest//python/python_api.html#module-xgboost.sklearn  
# MAGIC 
# MAGIC XGBoost is one of the most popular and widely used algorithms today because it is simply so powerful.
# MAGIC 
# MAGIC + It is similar to Gradient Boost but has a few extra features that make it that much stronger.
# MAGIC + Newton Boosting — Provides a direct route to the minima than gradient descent, making it much faster.
# MAGIC + An extra randomization parameter — reduces the correlation between trees, ultimately improving the strength of the ensemble.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC import xgboost
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
# MAGIC                                         silent=False,
# MAGIC                                         use_label_encoder=False,
# MAGIC                                         random_state=0).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_xgb = clf_eval(clf_xgb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Light GBM: 
# MAGIC 
# MAGIC It is another type of boosting algorithm that has shown to be faster and sometimes more accurate than XGBoost.
# MAGIC 
# MAGIC + It uses a unique technique called Gradient-based One-side sampling (GOSS) to filter out the data instances to find a split value.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC import lightgbm as lgb
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
# MAGIC #### Regularized Greedy Forest
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/  
# MAGIC https://github.com/fukatani/rgf_python  
# MAGIC https://github.com/MLWave/RGF-sklearn -- another implementation  
# MAGIC https://github.com/RGF-team/rgf_python -- another implementation  
# MAGIC 
# MAGIC 
# MAGIC In Boosting algorithms, each classifier/regressor is trained on data, taking into account the previous classifiers’/regressors’ success. After each training step, the weights are redistributed. Mis-classified data increases its weights to emphasize the most difficult cases. In this way, subsequent learners will focus on them during their training. However, the boosting methods simply treat the decision tree base learner as a black box and it does not take advantage of the tree structure itself.  In a sense, boosting does a partial corrective step to the model at each iteration. In contrast, RGF performs 2 steps:
# MAGIC 
# MAGIC + Finds the one step structural change to the current forest to obtain the new forest that minimises the loss function (e.g. Least squares or logloss)
# MAGIC + Adjusts the leaf weights for the entire forest to minimize the loss function

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from rgf.sklearn import RGFClassifier, FastRGFClassifier
# MAGIC 
# MAGIC clf_rgf = RGFClassifier(max_leaf=100,
# MAGIC                         algorithm="RGF_Sib",
# MAGIC                         test_interval=60,
# MAGIC                         verbose=False,).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_rgf = clf_eval(clf_rgf, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Bayesian Classifiers](http://scikit-learn.org/stable/modules/naive_bayes.html)  
# MAGIC 
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html  
# MAGIC http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html  
# MAGIC 
# MAGIC Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression, which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.
# MAGIC 
# MAGIC + As the name specifies, this algorithm is entirely based on Bayes's theorem. Bayes’ theorem says we can calculate the probability of a piece of data belonging to a given class if prior knowledge is given.  
# MAGIC + P(class|data) = (P(data|class) * P(class)) / P(data)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.naive_bayes import GaussianNB
# MAGIC 
# MAGIC clf_gnb = GaussianNB().fit(X_train, y_train)
# MAGIC 
# MAGIC roc_gnb = clf_eval(clf_gnb, X_test, y_test)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.naive_bayes import BernoulliNB
# MAGIC 
# MAGIC clf_bnb = BernoulliNB(alpha=0.20, 
# MAGIC                      binarize=0.0, 
# MAGIC                      fit_prior=True, 
# MAGIC                      class_prior=None).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_bnb = clf_eval(clf_bnb, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gaussian Processes
# MAGIC 
# MAGIC In the simple linear regression setting, we have a dependent variable y that we assume can be modeled as a function of an independent variable x, i.e. y=f(x)+ϵ (where ϵ is the irreducible error) but we assume further that the function f defines a linear relationship and so we are trying to find the parameters θ0  and θ1 which define the intercept and slope of the line respectively, i.e. y=θ0+θ1x+ϵ. Bayesian linear regression provides a probabilistic approach to this by finding a distribution over the parameters that gets updated whenever new data points are observed. The GP approach, in contrast, is a non-parametric approach, in that it finds a distribution over the possible functions f(x) that are consistent with the observed data. As with all Bayesian methods it begins with a prior distribution and updates this as data points are observed, producing the posterior distribution over functions.

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.gaussian_process import GaussianProcessClassifier
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
# MAGIC ### Neural Networks
# MAGIC 
# MAGIC #### Tensorflow and Keras

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC import tensorflow as tf
# MAGIC 
# MAGIC X_train_kr = X_train.astype('float32') 
# MAGIC X_test_kr = X_test.astype('float32')
# MAGIC y_train_kr = y_train
# MAGIC y_test_kr = y_test
# MAGIC 
# MAGIC print(X_train_kr.shape[0], 'train samples')
# MAGIC print(X_test_kr.shape[0], 'test samples')
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC batch_size = int(len(X_train_kr)/25)
# MAGIC num_classes = 2
# MAGIC epochs = 11
# MAGIC np.random.seed(0)
# MAGIC 
# MAGIC model = tf.keras.Sequential()
# MAGIC model.add(tf.keras.layers.Dense(input_dim=X_train_kr.shape[1], units=100, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=100, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=100, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=24, activation='relu'))
# MAGIC model.add(tf.keras.layers.Dropout(0.25))
# MAGIC model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #kernel_initializer='uniform',
# MAGIC model.summary()
# MAGIC 
# MAGIC #opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
# MAGIC opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.01, epsilon=1e-07, centered=False, name='RMSprop')
# MAGIC 
# MAGIC 
# MAGIC model.compile(loss='binary_crossentropy',
# MAGIC               optimizer=opt,
# MAGIC               metrics=['binary_accuracy'])
# MAGIC 
# MAGIC history = model.fit(X_train_kr, y_train_kr,
# MAGIC                     batch_size=batch_size,
# MAGIC                     epochs=epochs,
# MAGIC                     verbose=0,
# MAGIC                     shuffle=False,
# MAGIC                     validation_data=(X_test_kr, y_test_kr))
# MAGIC 
# MAGIC score = model.evaluate(X_test_kr, y_test_kr, verbose=0)
# MAGIC print('Test loss:', score[0])
# MAGIC print('Test accuracy:', score[1])
# MAGIC 
# MAGIC #WARNING:tensorflow:From <timed exec>:50: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated 
# MAGIC # and will be removed after 2021-01-01. Instructions for updating:
# MAGIC # Please use instead: `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   
# MAGIC #(e.g. if it uses a `softmax` last-layer activation).
# MAGIC #`(model.predict(x) > 0.5).astype("int32")`, if your model does binary classification (e.g. if it uses a `sigmoid` last-layer activation).
# MAGIC 
# MAGIC #y_pred = model.predict_classes(X_test_kr) 
# MAGIC y_pred = (model.predict(X_test_kr) > 0.5).astype("int32")
# MAGIC 
# MAGIC clf_matrix = confusion_matrix(y_test_kr, y_pred)
# MAGIC print('Classification Report')
# MAGIC print(classification_report(y_test_kr, y_pred, target_names=['Perished', 'Survived']))
# MAGIC print('ROC Score: {}'.format(roc_auc_score(y_test_kr, y_pred)))
# MAGIC print('Accuracy Score: {}'.format(accuracy_score(y_test_kr, y_pred)))
# MAGIC print('Average Precision Score: {}'.format(average_precision_score(y_test_kr, y_pred)))
# MAGIC print('f1 Score: {}'.format(f1_score(y_test_kr, y_pred)))
# MAGIC plot_confusion_matrix(clf_matrix, classes=['Perished', 'Survived'])
# MAGIC roc_keras = roc_auc_score(y_test_kr, y_pred)

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
# MAGIC ### Genetic Algorithms / AutoML
# MAGIC 
# MAGIC https://github.com/rhiever/tpot  
# MAGIC https://github.com/rhiever/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from tpot import TPOTClassifier
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
# MAGIC 
# MAGIC #clf_tpot.export('tpot_exported_pipeline.py')

# COMMAND ----------

# %load tpot_exported_pipeline.py

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Voting / Stacking](http://scikit-learn.org/stable/modules/ensemble.html#votingclassifier)  
# MAGIC 
# MAGIC [article](https://medium.com/@satyam-kumar/use-voting-classifier-to-improve-the-performance-of-your-ml-model-805345f9de0e)  
# MAGIC 
# MAGIC Voting algorithms are simple strategies, where you aglomerate results of classifiers' decisions by for example taking the class which appears in most cases. 
# MAGIC 
# MAGIC Stacking/grading strategies are generalizations of this concept. Instead of simply saying "ok, I have a scheme v, which I will use to select the best answer among my k classifiers" you create another abstraction layer, where you actually learn to predict the correct label having k votes.  

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC from sklearn.ensemble import VotingClassifier 
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
# MAGIC                                             #('clf_autoskl', clf_autoskl),
# MAGIC                                             #('clf_tpot', clf_tpot),
# MAGIC                                             ],
# MAGIC                                 voting='hard',
# MAGIC                                 weights=[1,1,1,1,1,1,1,2,1,2,1]).fit(X_train, y_train)
# MAGIC 
# MAGIC roc_ens = clf_eval(ensemble, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plotting the results:  

# COMMAND ----------

dic_results = {'SVM': roc_svc2,
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
               'GaussianNB': roc_gnb,
               'BernouNB': roc_bnb,
               'RGF': roc_rgf,
               'XGBoost':roc_xgb,
               'Knn':roc_knn,
               'Tensorflow': roc_keras,
               'Bagging': roc_bgc,
               'Voting': roc_ens,
               'Tpot': roc_tpot,
               'RGF': roc_rgf,
               #'AutoSKL': roc_autoskl,
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
ax.set_ylim(ymin=0.65,ymax = 0.85)
ax.set_title("Classificators' performance")
ax.set_xticks(ind + width/2.)
ax.set_xticklabels(list(zip(*tup_results))[0], rotation=45)

plt.show()

# COMMAND ----------

features = df_train2.iloc[:,2:].columns
df_fi = pd.DataFrame({'clf_lr': (abs(clf_lr.coef_[0])/sum(abs(clf_lr.coef_[0]))),    #a sort of adaptation
                      'clf_rf':clf_rf.feature_importances_,
                      'clf_xgb':clf_xgb.feature_importances_,
                      'clf_etc':clf_etc.feature_importances_,
                      'clf_abc':clf_abc.feature_importances_,
                      'clf_bgc':clf_bgc.estimators_[0].feature_importances_,
                      'clf_gbc':clf_gbc.feature_importances_,
                      'clf_lgb':clf_lgb.feature_importances_ / sum(clf_lgb.feature_importances_),
                     },
                      index=features)
df_fi['mean_importance'] = df_fi.mean(axis=1)
df_fi.sort_values(['mean_importance'], ascending=False, inplace=True)
df_fi

# COMMAND ----------

# MAGIC %md
# MAGIC Choosing the best classifier and training with all training data:

# COMMAND ----------

output = clf_rgf.predict(X_test)
print(output[10:20])
print()

output_prob = clf_rgf.predict_proba(X_test)
print(output_prob[10:20])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examining the ROC Curve

# COMMAND ----------

# calculate the fpr and tpr for all thresholds of the classification
probs = clf_rgf.predict_proba(X_train)
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
