# Databricks notebook source
# MAGIC %md
# MAGIC ### Topic Modelling and Clustering documents  

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook is based on [this](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/), [this](http://www.brandonrose.org) and [this](http://www.brandonrose.org/top100) posts. [This](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730) one added later.

# COMMAND ----------

# MAGIC %md
# MAGIC !pip install gensim 
# MAGIC !pip install mpld3
# MAGIC !pip install pyldavis

# COMMAND ----------

import re
import os
import codecs
import string
import numpy as np
import pandas as pd
import gensim

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import feature_extraction
import mpld3

import pyLDAvis
import pyLDAvis.gensim_models

from IPython.display import display, Image
from IPython.core.interactiveshell import InteractiveShell

%matplotlib inline
#%matplotlib notebook

# COMMAND ----------

# MAGIC %md
# MAGIC Specifying the path to the files  

# COMMAND ----------

datapath = "./data/doccluster/"
datapath = "./data/doccluster/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 - Clustering documents [example](https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create some documents.  

# COMMAND ----------

raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
                 "I am the barber who cuts everyone's hair who doesn't cut their own.",
                 "Legend has it that the mind is a mad monkey.",
                 "I make my own fun."]

print("Number of documents:",len(raw_documents))

# COMMAND ----------

# MAGIC %md
# MAGIC We will use NLTK to tokenize.  
# MAGIC A document will now be a list of tokens.  

# COMMAND ----------

nltk.download('punkt')

# COMMAND ----------

gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]

print(gen_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC We will create a dictionary from a list of documents.  
# MAGIC A dictionary maps every word to a number.  

# COMMAND ----------

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])
print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
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
s = 0
for i in corpus:
    s += len(i)
print(s)

# COMMAND ----------

for d in tf_idf[corpus]:
    print(d)

# COMMAND ----------

sims = gensim.similarities.Similarity(outputs,tf_idf[corpus],num_features=len(dictionary))
print(sims)
print(type(sims))

# COMMAND ----------

# MAGIC %md
# MAGIC Now create a query document and convert it to tf-idf.  

# COMMAND ----------

query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)

query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)

query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

# COMMAND ----------

# MAGIC %md
# MAGIC We show an array of document similarities to query.  
# MAGIC We see that the second document is the most similar with the overlapping of socks and force.

# COMMAND ----------

sims[query_doc_tf_idf]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 - [Topic Modeling](http://www.cs.columbia.edu/~blei/topicmodeling.html)  
# MAGIC 
# MAGIC Analytics Industry is all about obtaining the “Information” from the data. With the growing amount of data in recent years, that too mostly unstructured, it’s difficult to obtain the relevant and desired information. But, technology has developed some powerful methods which can be used to mine through the data and fetch the information that we are looking for.  
# MAGIC 
# MAGIC One such technique in the field of text mining is Topic Modelling. As the name suggests, it is a process to automatically identify topics present in a text object and to derive hidden patterns exhibited by a text corpus. Thus, assisting better decision making.  
# MAGIC 
# MAGIC Topic Modelling is different from rule-based text mining approaches that use regular expressions or dictionary based keyword searching techniques. It is an unsupervised approach used for finding and observing the bunch of words (called “topics”) in large clusters of texts.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Topics can be defined as “a repeating pattern of co-occurring terms in a corpus”. A good topic model should result in – “health”, “doctor”, “patient”, “hospital” for a topic – Healthcare, and “farm”, “crops”, “wheat” for a topic – “Farming”.  
# MAGIC 
# MAGIC Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection. For Example – New York Times are using topic models to boost their user – article recommendation engines. Various professionals are using topic models for recruitment industries where they aim to extract latent features of job descriptions and map them to right candidates. They are being used to organize large datasets of emails, customer reviews, and user social media profiles.  

# COMMAND ----------

#Image(url='https://www.analyticsvidhya.com/wp-content/uploads/2016/08/Modeling1.png')
Image(filename='../datasets/Figs/Modeling1.png')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 - Latent Dirichlet Allocation for Topic Modeling  
# MAGIC 
# MAGIC There are many approaches for obtaining topics from a text such as – Term Frequency and Inverse Document Frequency. NonNegative Matrix Factorization techniques. Latent Dirichlet Allocation is the most popular topic modeling technique and in this article, we will discuss the same.  
# MAGIC 
# MAGIC LDA assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.  
# MAGIC 
# MAGIC LDA is a matrix factorization technique. In vector space, any corpus (collection of documents) can be represented as a document-term matrix. The following matrix shows a corpus of N documents D1, D2, D3 … Dn and vocabulary size of M words W1,W2 .. Wn. The value of i,j cell gives the frequency count of word Wj in Document Di.  

# COMMAND ----------

#Image(url='https://www.analyticsvidhya.com/wp-content/uploads/2016/08/Modeling2.png')
Image(filename='../datasets/Figs/Modeling2.png')

# COMMAND ----------

# MAGIC %md
# MAGIC LDA converts this Document-Term Matrix into two lower dimensional matrices – M1 and M2.
# MAGIC M1 is a document-topics matrix and M2 is a topic – terms matrix with dimensions (N,  K) and (K, M) respectively, where N is the number of documents, K is the number of topics and M is the vocabulary size.  

# COMMAND ----------

Image(filename='../datasets/Figs/modeling3.png')

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that these two matrices already provides topic word and document topic distributions, However, these distribution needs to be improved, which is the main aim of LDA. LDA makes use of sampling techniques in order to improve these matrices.  
# MAGIC 
# MAGIC It Iterates through each word “w” for each document “d” and tries to adjust the current topic – word assignment with a new assignment. A new topic “k” is assigned to word “w” with a probability P which is a product of two probabilities p1 and p2.  
# MAGIC 
# MAGIC For every topic, two probabilities p1 and p2 are calculated. P1 – p(topic t / document d) = the proportion of words in document d that are currently assigned to topic t. P2 – p(word w / topic t) = the proportion of assignments to topic t over all documents that come from this word w.  
# MAGIC 
# MAGIC The current topic – word assignment is updated with a new topic with the probability, product of p1 and p2 . In this step, the model assumes that all the existing word – topic assignments except the current word are correct. This is essentially the probability that topic t generated word w, so it makes sense to adjust the current word’s topic with new probability.  
# MAGIC 
# MAGIC After a number of iterations, a steady state is achieved where the document topic and topic term distributions are fairly good. This is the convergence point of LDA.  
# MAGIC 
# MAGIC  
# MAGIC Parameters of LDA  
# MAGIC 
# MAGIC Alpha and Beta Hyperparameters – alpha represents document-topic density and Beta represents topic-word density. Higher the value of alpha, documents are composed of more topics and lower the value of alpha, documents contain fewer topics. On the other hand, higher the beta, topics are composed of a large number of words in the corpus, and with the lower value of beta, they are composed of few words.  
# MAGIC 
# MAGIC Number of Topics – Number of topics to be extracted from the corpus. Researchers have developed approaches to obtain an optimal number of topics by using Kullback Leibler Divergence Score. I will not discuss this in detail, as it is too mathematical. For understanding, one can refer to this[1] original paper on the use of KL divergence.  
# MAGIC 
# MAGIC Number of Topic Terms – Number of terms composed in a single topic. It is generally decided according to the requirement. If the problem statement talks about extracting themes or concepts, it is recommended to choose a higher number, if problem statement talks about extracting features or terms, a low number is recommended.  
# MAGIC 
# MAGIC Number of Iterations / passes – Maximum number of iterations allowed to LDA algorithm for convergence.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 - Topic Modeling Example 1 - Gensim

# COMMAND ----------

# Here are the sample documents combining together to form a corpus.

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

# COMMAND ----------

# MAGIC %md
# MAGIC Cleaning and Preprocessing
# MAGIC 
# MAGIC Cleaning is an important step before any text mining task, in this step, we will remove the punctuations, stopwords and normalize the corpus.  

# COMMAND ----------

nltk.download('stopwords')
nltk.download('wordnet')

# COMMAND ----------

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]   

# COMMAND ----------

# MAGIC %md
# MAGIC Preparing Document-Term Matrix
# MAGIC 
# MAGIC All the text documents combined is known as the corpus. To run any mathematical model on text corpus, it is a good practice to convert it into a matrix representation. LDA model looks for repeating term patterns in the entire DT matrix. Python provides many great libraries for text mining practices, “gensim” is one such clean and beautiful library to handle text data. It is scalable, robust and efficient. Following code shows how to convert a corpus into a document-term matrix.  

# COMMAND ----------

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# COMMAND ----------

# MAGIC %md
# MAGIC Running LDA Model  
# MAGIC 
# MAGIC Next step is to create an object for LDA model and train it on Document-Term matrix. The training also requires few parameters as input which are explained in the above section. The gensim module allows both LDA model estimation from a training corpus and inference of topic distribution on new, unseen documents.

# COMMAND ----------

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# COMMAND ----------

print(ldamodel.print_topics(num_topics=3, num_words=3))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 - Topic Modeling Example 2 (Scikit Learn)

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

# COMMAND ----------

# MAGIC %md
# MAGIC Gensim is an awesome library and scales really well to large text corpuses. Gensim, however does not include Non-negative Matrix Factorization (NMF), which can also be used to find topics in text. The mathematical basis underpinning NMF is quite different from LDA. NMF sometimes produces more meaningful topics for smaller datasets. NMF has been included in Scikit Learn for quite a while but LDA has only recently (late 2015) been included. The great thing about using Scikit Learn is that it brings API consistency which makes it almost trivial to perform Topic Modeling using both LDA and NMF. Scikit Learn also includes seeding options for NMF which greatly helps with algorithm convergence and offers both online and batch variants of LDA.

# COMMAND ----------

display(Image(os.path.join('../datasets/','Figs', 'nmf.png'), width=700))

# COMMAND ----------

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# COMMAND ----------

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

no_features = 1000

# COMMAND ----------

# MAGIC %md
# MAGIC The creation of the bag of words matrix is very easy in Scikit Learn — all the heavy lifting is done by the feature extraction functionality provided for text datasets. A tf-idf transformer is applied to the bag of words matrix that NMF must process with the TfidfVectorizer. LDA on the other hand, being a probabilistic graphical model (i.e. dealing with probabilities) only requires raw counts, so a CountVectorizer is used. Stop words are removed and the number of terms included in the bag of words matrix is restricted to the top 1000.

# COMMAND ----------

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# COMMAND ----------

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

# COMMAND ----------

# MAGIC %md
# MAGIC As mentioned previously the algorithms are not able to automatically determine the number of topics and this value must be set when running the algorithm. Comprehensive documentation on available parameters is available for both NMF and LDA. Initialising the W and H matrices in NMF with ‘nndsvd’ rather than random initialisation improves the time it takes for NMF to converge. LDA can also be set to run in either batch or online mode.

# COMMAND ----------

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

# COMMAND ----------

# MAGIC %md
# MAGIC Displaying and Evaluating Topics
# MAGIC 
# MAGIC The structure of the resulting matrices returned by both NMF and LDA is the same and the Scikit Learn interface to access the returned matrices is also the same. This is great and allows for a common Python method that is able to display the top words in a topic. Topics are not labeled by the algorithm — a numeric index is assigned.

# COMMAND ----------

# MAGIC %md
# MAGIC The derived topics from NMF and LDA are displayed below. From the NMF derived topics, Topic 0 and 8 don’t seem to be about anything in particular but the other topics can be interpreted based upon there top words. LDA for the 20 Newsgroups dataset produces 2 topics with noisy data (i.e., Topic 4 and 7) and also some topics that are hard to interpret (i.e., Topic 3 and Topic 9). I’d say the NMF was able to find more meaningful topics in the 20 Newsgroups dataset.

# COMMAND ----------

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)

# COMMAND ----------

display_topics(lda, tf_feature_names, no_top_words)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 - Tips to improve results of topic modeling
# MAGIC 
# MAGIC The results of topic models are completely dependent on the features (terms) present in the corpus. The corpus is represented as document term matrix, which in general is very sparse in nature. Reducing the dimensionality of the matrix can improve the results of topic modelling. Based on my practical experience, there are few approaches which do the trick.  
# MAGIC 
# MAGIC 1. Frequency Filter – Arrange every term according to its frequency. Terms with higher frequencies are more likely to appear in the results as compared ones with low frequency. The low frequency terms are essentially weak features of the corpus, hence it is a good practice to get rid of all those weak features. An exploratory analysis of terms and their frequency can help to decide what frequency value should be considered as the threshold.  
# MAGIC 
# MAGIC 2. Part of Speech Tag Filter – POS tag filter is more about the context of the features than frequencies of features. Topic Modelling tries to map out the recurring patterns of terms into topics. However, every term might not be equally important contextually. For example, POS tag IN contain terms such as – “within”, “upon”, “except”. “CD” contains – “one”,”two”, “hundred” etc. “MD” contains “may”, “must” etc. These terms are the supporting words of a language and can be removed by studying their post tags.  
# MAGIC 
# MAGIC 3. Batch Wise LDA –In order to retrieve most important topic terms, a corpus can be divided into batches of fixed sizes. Running LDA multiple times on these batches will provide different results, however, the best topic terms will be the intersection of all batches.  
# MAGIC 
# MAGIC #### 2.5 - Topic Modelling for Feature Selection
# MAGIC 
# MAGIC Sometimes LDA can also be used as feature selection technique. Take an example of text classification problem where the training data contain category wise documents. If LDA is running on sets of category wise documents. Followed by removing common topic terms across the results of different categories will give the best features for a category.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 - Clustering and Topic Modeling applied to film synopses

# COMMAND ----------

# MAGIC %md
# MAGIC In this guide, I will explain how to cluster a set of documents using Python. My motivating example is to identify the latent structures within the synopses of the top 100 films of all time (per an IMDB list). 
# MAGIC 
# MAGIC It will cover:
# MAGIC 
# MAGIC <ul>
# MAGIC <li> tokenizing and stemming each synopsis
# MAGIC <li> transforming the corpus into vector space using [tf-idf](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)
# MAGIC <li> calculating cosine distance between each document as a measure of similarity
# MAGIC <li> clustering the documents using the [k-means algorithm](http://en.wikipedia.org/wiki/K-means_clustering)
# MAGIC <li> using [multidimensional scaling](http://en.wikipedia.org/wiki/Multidimensional_scaling) to reduce dimensionality within the corpus
# MAGIC <li> plotting the clustering output using [matplotlib](http://matplotlib.org/) and [mpld3](http://mpld3.github.io/)
# MAGIC <li> conducting a hierarchical clustering on the corpus using [Ward clustering](http://en.wikipedia.org/wiki/Ward%27s_method)
# MAGIC <li> plotting a Ward dendrogram
# MAGIC <li> topic modeling using [Latent Dirichlet Allocation (LDA)](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 - Contents

# COMMAND ----------

# MAGIC %md
# MAGIC <ul>
# MAGIC <li>[Stopwords, stemming, and tokenization](#Stopwords,-stemming,-and-tokenizing)
# MAGIC <li>[Tf-idf and document similarity](#Tf-idf-and-document-similarity)
# MAGIC <li>[K-means clustering](#K-means-clustering)
# MAGIC <li>[Multidimensional scaling](#Multidimensional-scaling)
# MAGIC <li>[Visualizing document clusters](#Visualizing-document-clusters)
# MAGIC <li>[Hierarchical document clustering](#Hierarchical-document-clustering)
# MAGIC <li>[Latent Dirichlet Allocation (LDA)](#Latent-Dirichlet-Allocation)
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md
# MAGIC But first, I import everything I am going to need up front

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.1 - import three lists: titles, links and wikipedia synopses  

# COMMAND ----------

titles = open(os.path.join(datapath, 'title_list.txt')).read().split('\n')
#ensures that only the first 100 are read in
titles = titles[:100]
print(str(len(titles)) + ' titles')

print(titles[0:5])

# COMMAND ----------

links = open(os.path.join(datapath, 'link_list_imdb.txt')).read().split('\n')
links = links[:100]
print(str(len(links)) + ' links')

print(links[0:5])

# COMMAND ----------

synopses_wiki = open(os.path.join(datapath, 'synopses_list_wiki.txt'), encoding="utf8").read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_wiki.append(text)
synopses_wiki = synopses_clean_wiki

print(str(len(synopses_wiki)) + ' synopses')

print(synopses_wiki[0][0:627])

# COMMAND ----------

synopses_imdb = open(os.path.join(datapath,'synopses_list_imdb.txt'), encoding="utf8").read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]
synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

print(str(len(synopses_imdb)) + ' synopses')

print(synopses_imdb[0][0:627])

# COMMAND ----------

#Joining the two synopses sources

synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)
    
synopses[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.2 - Importing the genres

# COMMAND ----------

genres = open(os.path.join(datapath,'genres_list.txt')).read().split('\n')
genres = genres[:100]
print(str(len(genres)) + ' genres')

print(genres[0:5])

# COMMAND ----------

# generates index for each item in the corpora (in this case it's just rank) and I'll use this for scoring later
ranks = []

for i in range(0,len(titles)):
    ranks.append(i)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.3 - Stopwords, stemming, and tokenizing

# COMMAND ----------

# MAGIC %md
# MAGIC This section is focused on defining some functions to manipulate the synopses. First, I load [NLTK's](http://www.nltk.org/) list of English stop words. [Stop words](http://en.wikipedia.org/wiki/Stop_words) are words like "a", "the", or "in" which don't convey significant meaning. I'm sure there are much better explanations of this out there.

# COMMAND ----------

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# COMMAND ----------

# MAGIC %md
# MAGIC Next I import the [Snowball Stemmer](http://snowball.tartarus.org/) which is actually part of NLTK. [Stemming](http://en.wikipedia.org/wiki/Stemming) is just the process of breaking a word down into its root.

# COMMAND ----------

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Below I define two functions:
# MAGIC 
# MAGIC <ul>
# MAGIC <li> *tokenize_and_stem*: tokenizes (splits the synopsis into a list of its respective words (or tokens) and also stems each token <li> *tokenize_only*: tokenizes the synopsis only
# MAGIC </ul>
# MAGIC 
# MAGIC I use both these functions to create a dictionary which becomes important in case I want to use stems for an algorithm, but later convert stems back to their full words for presentation purposes. Guess what, I do want to do that!

# COMMAND ----------

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# COMMAND ----------

# MAGIC %md
# MAGIC Below I use my stemming/tokenizing and tokenizing functions to iterate over the list of synopses to create two vocabularies: one stemmed and one only tokenized. 

# COMMAND ----------

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

# COMMAND ----------

# MAGIC %md
# MAGIC Using these two lists, I create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column. The benefit of this is it provides an efficient way to look up a stem and return a full token. The downside here is that stems to tokens are one to many: the stem 'run' could be associated with 'ran', 'runs', 'running', etc. For my purposes this is fine--I'm perfectly happy returning the first token associated with the stem I need to look up.

# COMMAND ----------

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.4 - Tf-idf and document similarity

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='http://www.jiem.org/index.php/jiem/article/viewFile/293/252/2402' align='right' style="margin-left:10px">
# MAGIC 
# MAGIC Here, I define term frequency-inverse document frequency (tf-idf) vectorizer parameters and then convert the *synopses* list into a tf-idf matrix. 
# MAGIC 
# MAGIC To get a Tf-idf matrix, first count word occurrences by document. This is transformed into a document-term matrix (dtm). This is also just called a term frequency matrix. An example of a dtm is here at right.
# MAGIC 
# MAGIC Then apply the term frequency-inverse document frequency weighting: words that occur frequently within a document but not frequently within the corpus receive a higher weighting as these words are assumed to contain more meaning in relation to the document.
# MAGIC 
# MAGIC A couple things to note about the parameters I define below:
# MAGIC 
# MAGIC <ul>
# MAGIC <li> max_df: this is the maximum frequency within the documents a given feature can have to be used in the tfi-idf matrix. If the term is in greater than 80% of the documents it probably cares little meanining (in the context of film synopses)
# MAGIC <li> min_idf: this could be an integer (e.g. 5) and the term would have to be in at least 5 of the documents to be considered. Here I pass 0.2; the term must be in at least 20% of the document. I found that if I allowed a lower min_df I ended up basing clustering on names--for example "Michael" or "Tom" are names found in several of the movies and the synopses use these names frequently, but the names carry no real meaning.
# MAGIC <li> ngram_range: this just means I'll look at unigrams, bigrams and trigrams. See [n-grams](http://en.wikipedia.org/wiki/N-gram)
# MAGIC </ul>

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, 
                                   max_features=200000,
                                   min_df=0.2, 
                                   stop_words='english',
                                   use_idf=True, 
                                   tokenizer=tokenize_and_stem, 
                                   ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

print(tfidf_matrix.shape)

# COMMAND ----------

terms = tfidf_vectorizer.get_feature_names()

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.5 - K-means clustering

# COMMAND ----------

# MAGIC %md
# MAGIC Now onto the fun part. Using the tf-idf matrix, you can run a slew of clustering algorithms to better understand the hidden structure within the synopses. I first chose [k-means](http://en.wikipedia.org/wiki/K-means_clustering). K-means initializes with a pre-determined number of clusters (I chose 5). Each observation is assigned to a cluster (cluster assignment) so as to minimize the within cluster sum of squares. Next, the mean of the clustered observations is calculated and used as the new cluster centroid. Then, observations are reassigned to clusters and  centroids recalculated in an iterative process until the algorithm reaches convergence.
# MAGIC 
# MAGIC I found it took several runs for the algorithm to converge a global optimum as k-means is susceptible to reaching local optima. 

# COMMAND ----------

from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# COMMAND ----------

#from sklearn.externals import joblib
import joblib

joblib.dump(km, os.path.join(outputs, 'doc_cluster.pkl'))
km = joblib.load(os.path.join(outputs, 'doc_cluster.pkl'))
clusters = km.labels_.tolist()

# COMMAND ----------

films = {'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres}
frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])

# COMMAND ----------

frame.head()

# COMMAND ----------

frame['cluster'].value_counts()

# COMMAND ----------

grouped = frame['rank'].groupby(frame['cluster'])

grouped.mean()

# COMMAND ----------

from __future__ import print_function

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print(f"Cluster {i} words:", end='')
    for ind in order_centroids[i, :6]:
        print(f" {vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')}", end=',')
    print()
    print()
    print(f"Cluster {i} titles:", end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(f' {title},', end='')
    print()
    print()

# COMMAND ----------

#This is purely to help export tables to html and to correct for my 0 start rank (so that Godfather is 1, not 0)
frame['Rank'] = frame['rank'] + 1
frame['Title'] = frame['title']

# COMMAND ----------

#export tables to HTML
html_output = frame[['Rank', 'Title']].loc[frame['cluster'] == 1].to_html(index=False)

from IPython.core.display import display, HTML
display(HTML(html_output))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.6 - Multidimensional scaling

# COMMAND ----------

from sklearn.manifold import MDS

# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

# COMMAND ----------

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.7 - Visualizing document clusters

# COMMAND ----------

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}

# COMMAND ----------

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False)
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  

plt.show()

#Saving the Fig
plt.savefig(os.path.join(outputs,'clusters_small_noaxes.png'), dpi=200)

# COMMAND ----------

# MAGIC %md
# MAGIC The clustering plot looks great, but it would be better without overlapping labels. We are going to use D3.js (http://d3js.org/), a browser based/javascript interactive. We will use a matplotlib D3 wrapper called mpld3 (https://mpld3.github.io/). Mpld3 basically let's you use matplotlib syntax to create web interactives. It has a really easy, high-level API for adding tooltips on mouse hover, which is what I am interested in.
# MAGIC 
# MAGIC It also has some nice functionality for zooming and panning. The below javascript snippet basicaly defines a custom location for where the zoom/pan toggle resides. Don't worry about it too much and you actually don't need to use it, but it helped for formatting purposes when exporting to the web later. The only thing you might want to change is the x and y attr for the position of the toolbar.

# COMMAND ----------

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

# COMMAND ----------

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }
"""

# Plot 
fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# COMMAND ----------

#uncomment the below to export to html
#os.chdir(outputs)
#html = mpld3.fig_to_html(fig)
#print(html)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.8 - Hierarchical document clustering

# COMMAND ----------

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.tight_layout() #show plot with tight layout
plt.savefig(os.path.join(outputs,'ward_clusters.png'), dpi=200)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.9 - Latent Dirichlet Allocation

# COMMAND ----------

#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
import string
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

# COMMAND ----------

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# COMMAND ----------

#Latent Dirichlet Allocation implementation with Gensim
from gensim import corpora, models, similarities 

#remove proper names
preprocess = [strip_proppers(doc) for doc in synopses]

%time tokenized_text = [tokenize_and_stem(text) for text in preprocess]
%time texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

# COMMAND ----------

#print(len([word for word in texts[0] if word not in stopwords]))
print(len(texts[0]))

# COMMAND ----------

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in texts]

# COMMAND ----------

len(corpus)

# COMMAND ----------

# MAGIC %time lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

# COMMAND ----------

print(lda[corpus[0]])

# COMMAND ----------

topics = lda.print_topics(5, num_words=20)

# COMMAND ----------

lda.show_topics()

# COMMAND ----------

topics_matrix = lda.show_topics(formatted=False, num_words=20)

# COMMAND ----------

topics_matrix[0]

# COMMAND ----------

pyLDAvis.enable_notebook()
pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
