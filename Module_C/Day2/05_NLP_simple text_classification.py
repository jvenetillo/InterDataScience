# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction to text classification
# MAGIC 
# MAGIC In this task, you will get a sense of using Python to do **text processing**. The dataset that we have is a public collection of SMS messages that have been collected for mobile phone spam reserach. <br>
# MAGIC The **goal** is: based on the text in the sms message to predict whether the message is a spam or not. We will illustrate basic text classification approchaches on [
# MAGIC SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) data set. 
# MAGIC The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
# MAGIC 
# MAGIC 
# MAGIC ## Dataset Information
# MAGIC We have the same set of data in two file formats: csv and txt file. Baesd on your preference, you can choose one file format to work with 😊. <br>
# MAGIC The dataset consists of a collection of different sms messages, with each line consisted of one label (either ham or spam) and the raw message. 
# MAGIC 
# MAGIC **See some examples below:** <br>
# MAGIC ham What you doing?how are you? <br>
# MAGIC ham Ok lar... Joking wif u oni... <br>
# MAGIC ham dun say so early hor... U c already then say... <br>
# MAGIC ham MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*<br>
# MAGIC ham Siva is in hostel aha:-.<br>
# MAGIC ham Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.<br>
# MAGIC spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop<br>
# MAGIC spam Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B<br>
# MAGIC spam URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU<br>
# MAGIC 
# MAGIC Note: the messages are not chronologically sorted.
# MAGIC 
# MAGIC Dataset Citation:http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
# MAGIC The SMS Spam Collection has been created by [Tiago A. Almeida](http://dcomp.sor.ufscar.br/talmeida/) and [José María Gómez Hidalgo](http://www.esp.uem.es/jmgomez).

# COMMAND ----------

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from zipfile import ZipFile

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disabling MLFlow autologging

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading dataset

# COMMAND ----------

df = pd.read_csv('./data/spam.csv', sep='\t')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation of train and test data sets
# MAGIC 
# MAGIC Separate and rename target values.

# COMMAND ----------

data = df['Text']
target = df['Target'].replace('ham', 1).replace('spam', 0)
names = ['spam', 'ham']
print("\n", data[:5])
print("\n", target[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC Shuffle the data and split it to train and test parts.

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing
# MAGIC 
# MAGIC Tokenize the texts. Experiment with various tokenizers from the [NLTK](http://www.nltk.org/api/nltk.tokenize.html) library.

# COMMAND ----------

from nltk.tokenize.casual import casual_tokenize

# COMMAND ----------

sms = data[4]
print(sms)

# COMMAND ----------

tokenizer = lambda text: casual_tokenize(text, preserve_case=False)
print(tokenizer(sms))

# COMMAND ----------

# MAGIC %md
# MAGIC Convert tokens to their stems. Experiment with stemmers and lemmatizers from the [NLTK](http://www.nltk.org/api/nltk.stem.html) library.

# COMMAND ----------

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# COMMAND ----------

stem_tokenizer = lambda text: [stemmer.stem(w) for w in tokenizer(text)]
print (stem_tokenizer(sms))

# COMMAND ----------

# MAGIC %md
# MAGIC Fit a [vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) which converts texts to count vectors.

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=stem_tokenizer, 
                             #token_pattern='(?u)\b\w\w+\b', 
                             stop_words="english",
                             strip_accents=None, 
                             lowercase=True, 
                             preprocessor=None, 
                             ngram_range=(1, 1), 
                             analyzer='word', 
                             max_df=1.0, 
                             min_df=1, 
                             max_features=None, 
                             vocabulary=None,
                             )

# COMMAND ----------

vectorizer.fit(X_train)
print (vectorizer.transform([sms]))

# COMMAND ----------

# MAGIC %md
# MAGIC Convert count vectors to TFIDF

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

# COMMAND ----------

tfidf_transformer.fit(vectorizer.transform(X_train))
print(tfidf_transformer.transform(vectorizer.transform([sms])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification
# MAGIC 
# MAGIC Train a classifier using the following models:
# MAGIC * [Logistic regression](http://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.LogisticRegression.html)
# MAGIC * [Gradient Boosted Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) (Experiment with different depths and number of trees)
# MAGIC * [Support Vector Machines](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) (experiment with different kernels)

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

# COMMAND ----------

clf_pipeline = Pipeline([('vec', vectorizer),
                         ('tfidf', tfidf_transformer),
                         ('lr', LogisticRegression()),
                         #('gbc', GradientBoostingClassifier(n_estimators=100, max_depth=4))
                         #('svm', svm.SVC(kernel='linear'))
                        ])
clf_pipeline.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC 
# MAGIC Compute common classification metrics and evaluate the models. Decide which model performs best on the given problem.

# COMMAND ----------

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# COMMAND ----------

y_pred = clf_pipeline.predict(X_test)

print ("Test accuracy: {:.5f}".format(accuracy_score(y_test, y_pred)))
print()
print(metrics.classification_report(y_test, y_pred, digits=4))

# COMMAND ----------

print(confusion_matrix(y_test, y_pred))
df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True,  fmt='.0f')

# COMMAND ----------


