# Databricks notebook source
# MAGIC %md
# MAGIC ### Document Similarity  

# COMMAND ----------

!pip install -U -q gensim 

# COMMAND ----------

import re
import os
import codecs
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import casual_tokenize

from sklearn import feature_extraction

import gensim

from IPython.display import display, Image
from IPython.core.interactiveshell import InteractiveShell

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create some documents.  

# COMMAND ----------

raw_documents = ["to open a bank account.",
                 "to pay in a cheque",
                 "to cash a cheque.",
                 "to transfer money.",
                 "to go into liquidation.",
                 "(for a company) to go into administration",
                 "to be in debt",
                 "to owe money (to someone)",
                 "to take out a loan",
                 "to insure something against fire/theft/accidental damage",
                 "to pay into a savings account/pension",
                 "to borrow money from someone",
                 "to pay by installments", 
                 "to lend money to someone",
                 "to invest in something/someone",
                 "to get a return on an investment",
                 "to change some money",
                 "to borrow money – to take money from someone that you will pay back later",
                 "to lend money – to give someone money that they will pay back later",
                ]



print("Number of documents:",len(raw_documents))

# COMMAND ----------

# MAGIC %md
# MAGIC We will use NLTK to tokenize.  
# MAGIC A document will now be a list of tokens.  

# COMMAND ----------

gen_docs = [[w.lower() for w in casual_tokenize(text)] for text in raw_documents]

print(gen_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC We will create a dictionary from a list of documents.  
# MAGIC A dictionary maps every word to a number.  

# COMMAND ----------

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])
print(dictionary.token2id['money'])
print("Number of words in dictionary:",len(dictionary))

# COMMAND ----------

for i in range(len(dictionary)):
    print(i, dictionary[i])

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will create a corpus. A corpus is a list of bags of words.  
# MAGIC A bag-of-words representation for a document just lists the number of times each word occurs in the document.  

# COMMAND ----------

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
for d in corpus:
    print(d)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we create a tf-idf model from the corpus.

# COMMAND ----------

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

# COMMAND ----------

for d in tf_idf[corpus]:
    print(d)

# COMMAND ----------

sims = gensim.similarities.Similarity("/tmp/",
                                      tf_idf[corpus],
                                      num_features=len(dictionary))
print(type(sims))
print(sims)

# COMMAND ----------

# MAGIC %md
# MAGIC Now create a query document and convert it to tf-idf.  

# COMMAND ----------

query_doc = [w.lower() for w in casual_tokenize("I’d like to open a savings account")]
print(query_doc)

# COMMAND ----------

query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)

# COMMAND ----------

# MAGIC %md
# MAGIC We show an array of document similarities to query.  

# COMMAND ----------

query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

# COMMAND ----------

sims[query_doc_tf_idf]

# COMMAND ----------

# MAGIC %md
# MAGIC Printing the most similar Document

# COMMAND ----------

print(raw_documents[np.argmax(sims[query_doc_tf_idf])])
