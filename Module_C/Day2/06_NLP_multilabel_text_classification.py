# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Label Text Classification

# COMMAND ----------

# MAGIC %md
# MAGIC Please check this [article](https://medium.com/analytics-vidhya/an-introduction-to-multi-label-text-classification-b1bcb7c7364c?sk=8a30075009552cfd4a7534663edaed7e) for a detailed explanation.

# COMMAND ----------

import os
import re
import time
from zipfile import ZipFile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# COMMAND ----------

begin = time.time()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Disabling MLFlow autologging

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading Data Files (directly from zip files)

# COMMAND ----------

with ZipFile(os.path.join("data", "topics", 'train.csv.zip'), 'r') as myzip:
    with myzip.open('train.csv') as myfile:
        train_df = pd.read_csv(myfile)
        
with ZipFile(os.path.join("data", "topics", 'test.csv.zip'), 'r') as myzip:
    with myzip.open('test.csv') as myfile:
        test_df = pd.read_csv(myfile)

# COMMAND ----------

train_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checking Missing Values and Data Types

# COMMAND ----------

train_df.info()

# COMMAND ----------

x = train_df.iloc[:,3:].sum()
rowsums = train_df.iloc[:,2:].sum(numeric_only=True, axis=1)
no_label_count = 0
for sum in rowsums.items():
    if sum == 0:
        no_label_count += 1

print("Total number of articles = ",len(train_df))
print("Total number of articles without label = ",no_label_count)
print("Total labels = ",x.sum())

# COMMAND ----------

print("Check for missing values in Train dataset")
print(train_df.isnull().sum().sum())

print("Check for missing values in Test dataset")
print(test_df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checking how many abstracts belongs to each category.

# COMMAND ----------

train_df.iloc[:,3:].apply(lambda x:x.sum(), axis=0).values

# COMMAND ----------

train_df.iloc[:,3:].apply(lambda x:x.sum(), axis=0).plot(kind="bar", figsize=(15,5))

# COMMAND ----------

# MAGIC %md
# MAGIC From the above plot its clear that "Quantitative biology" and "Quantitative Finance" have too few values compared to the other categories. This means that the data set is imbalanced.  
# MAGIC To make it balanced we can apply **resampling techniques**. The data set is small so we can try oversampling for these two classes.  
# MAGIC 
# MAGIC We will implement oversampling later. First we will try to build a basic classification model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checking the number of words in each category

# COMMAND ----------

total_word_count_in_each_category = []
for i in train_df.iloc[:,3:].columns:
    abstracts = train_df.where(train_df[i] == 1)[['ABSTRACT']]
    count = pd.Series(abstracts.values.flatten()).str.len().sum()
    total_word_count_in_each_category.append(count)

# COMMAND ----------

plt.figure(figsize=(15,5))
plt.bar(train_df.iloc[:,3:].columns,total_word_count_in_each_category)

# COMMAND ----------

# MAGIC %md
# MAGIC The word count is almost in the same proportion as the number of texts in each category.  
# MAGIC The only difference is statistics which has more words than mathematics even if the number of articles is more for mathematics.  
# MAGIC Now let's calculate the average number of words per category:

# COMMAND ----------

avg_abstract_len_for_each_category = []
for i in range(6):
    avg_abstract_len_for_each_category.append(total_word_count_in_each_category[i]/train_df.iloc[:,3:].apply(lambda x:x.sum(), axis=0).values[i])

# COMMAND ----------

plt.figure(figsize=(15,5))
plt.bar(train_df.iloc[:,3:].columns, avg_abstract_len_for_each_category)

# COMMAND ----------

# MAGIC %md
# MAGIC From the above plot its clear that articles of quantitative biology are the longest, and mathematics articles are the shortest.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preparing the Corpus  
# MAGIC Let's concatenate 'Title' and 'Abstract' and make it one big text.

# COMMAND ----------

train_df["text"] = train_df["TITLE"] + " " + train_df["ABSTRACT"]

# COMMAND ----------

# MAGIC %md
# MAGIC We drop the 'Title' and 'Abstract' columns as they are not needed anymore. 

# COMMAND ----------

train_df.drop(["TITLE","ABSTRACT"],axis=1,inplace=True)

# COMMAND ----------

train_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's make a function for train/test split as we will need this further.

# COMMAND ----------

def split(X,y,test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return (X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleaning the text

# COMMAND ----------

def clean_text(input_text):
    x = re.sub('[^\w]|_', ' ', input_text)  # only keep numbers and letters and spaces
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r'', x)  # remove non ascii texts
    x = [y for y in x.split(' ') if y] # remove empty words
    x = ['[number]' if y.isdigit() else y for y in x]
    cleaned_text =  ' '.join(x)
    return cleaned_text

# COMMAND ----------

train_df['cleaned_text'] = train_df['text'].apply(clean_text)

# COMMAND ----------

train_df.head()

# COMMAND ----------

train_df.cleaned_text[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Splitting our dataset using the previously defined function

# COMMAND ----------

X_train, X_test, y_train, y_test = split(train_df.loc[:,"cleaned_text"], train_df.iloc[:,1:7], 0.2)

# COMMAND ----------

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Now that our text is cleaned we will apply Tfidf on the text data to convert it into a matrix of numericals.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Changing text into numericals using Tfidf technique  

# COMMAND ----------

tfidf = TfidfVectorizer(min_df=3, 
                        max_features=10000, 
                        strip_accents="unicode", 
                        analyzer="word",
                        token_pattern=r"\w{1,}",
                        ngram_range=(1,2),
                        use_idf=1,
                        smooth_idf=1,
                        sublinear_tf=1,
                        stop_words="english")

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Fitting in the train data, transforming train and test data

# COMMAND ----------

tfidf.fit(list(X_train) + list(X_test))

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# COMMAND ----------

X_train_tfidf.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #### The target column is made up of 6 columns , so lets change it to one columns with all 6 different categories.

# COMMAND ----------

y_train_new = y_train.idxmax(axis=1)
y_test_new = y_test.idxmax(axis=1)

# COMMAND ----------

y_train_new.nunique(), y_test_new.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's apply a simple Logistic Regression model to classify.   
# MAGIC Apply a grid search to optimize the hyperparameters.

# COMMAND ----------

params={
    'C':[0.8,1, 1.3],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty':['l1', 'l2', 'elasticnet', 'none']
}

gs_lr = GridSearchCV(LogisticRegression(),
                     param_grid=params,
                     scoring='accuracy',
                     cv=3, n_jobs=-1)

gs_model = gs_lr.fit(X_train_tfidf, y_train_new)

# COMMAND ----------

gs_model.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating a LR classifier with the best parameters found

# COMMAND ----------

clf = LogisticRegression(C=gs_model.best_params_['C'],
                         solver=gs_model.best_params_['solver'],
                         penalty=gs_model.best_params_['penalty'],
                         n_jobs=-1)

clf.fit(X_train_tfidf, y_train_new)
clf.score(X_test_tfidf, y_test_new)

# COMMAND ----------

train_preds = clf.predict(X_train_tfidf)
test_preds = clf.predict(X_test_tfidf)
train_preds

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluating the Model

# COMMAND ----------

# MAGIC %md
# MAGIC Our data set is imbalanced and all the classes are equally important, so for this case a macro average F1 score would be the best.  
# MAGIC The confusion matrix would then give an overall good picture of every class's prediction.

# COMMAND ----------

print ("Test accuracy: {:.5f}".format(accuracy_score(y_test_new,test_preds)))
print()
print(metrics.classification_report(y_test_new,test_preds, digits=4))

# COMMAND ----------

print('train f1 score', f1_score(y_train_new, clf.predict(X_train_tfidf), average='macro'))
print('test f1 score', f1_score(y_test_new, clf.predict(X_test_tfidf), average='macro'))
print("train accuracy",accuracy_score(y_train_new,clf.predict(X_train_tfidf)))
print("test accuracy",accuracy_score(y_test_new,clf.predict(X_test_tfidf)))

# COMMAND ----------

# MAGIC %md
# MAGIC We got 79.93 accuracy using logistic regression and the macro average F1 score is 0.4557.

# COMMAND ----------

c_matrix = confusion_matrix(y_train_new,
                            clf.predict(X_train_tfidf))

c_matrix = pd.DataFrame(c_matrix,
                        columns=train_df.iloc[:,1:7].columns,
                        index=train_df.iloc[:,1:7].columns)

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.4)
sns.heatmap(c_matrix/np.sum(c_matrix), fmt="0.2%", annot=True, cmap="Blues", ax=ax)
ax.set_title("Confusion matrix ", fontsize=26)
ax.set_xlabel("Predicted", fontsize=26)
ax.set_ylabel("Actual", fontsize=26)

# COMMAND ----------

c_matrix = confusion_matrix(y_test_new,clf.predict(X_test_tfidf))

c_matrix = pd.DataFrame(c_matrix,
                        columns=train_df.iloc[:,1:7].columns,
                        index=train_df.iloc[:,1:7].columns)

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.4)
sns.heatmap(c_matrix/np.sum(c_matrix), fmt="0.2%", annot=True, cmap="Blues", ax=ax)
ax.set_title("Confusion matrix ", fontsize=26)
ax.set_xlabel("Predicted", fontsize=26)
ax.set_ylabel("Actual", fontsize=26)

# COMMAND ----------

# MAGIC %md
# MAGIC # References
# MAGIC 
# MAGIC * https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/  
# MAGIC * https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff  
# MAGIC * https://www.thepythoncode.com/article/text-classification-using-tensorflow-2-and-keras-in-python   
# MAGIC * https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles/code

# COMMAND ----------

print(f"Execution took: {((time.time() - begin)/60)} minutes")
