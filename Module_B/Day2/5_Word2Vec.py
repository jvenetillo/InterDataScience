# Databricks notebook source
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import re
import os
import sys
import pathlib
import multiprocessing
import urllib.request
import zipfile
import lxml.etree
import networkx as nx
from random import shuffle

import gensim 
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, MiniBatchKMeans


%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (20.0, 15.0)

# COMMAND ----------

import importlib
import logging
importlib.reload(logging)

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
#logging.root.setLevel(level=logging.INFO)
#logger = logging.getLogger()
#logger = logging.getLogger(program)
#logger.setLevel(logging.DEBUG)

# COMMAND ----------

# https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c
# https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models
# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

path_io_files = pathlib.Path('../datasets/Word2Vec/')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Evaluation Tests

# COMMAND ----------

questions = './data/questions-words.txt'

# COMMAND ----------

evals = open(questions).readlines()
num_sections = len([l for l in evals if l.startswith(':')])
print('total evaluation sentences: {} '.format(len(evals) - num_sections))

# COMMAND ----------

def w2v_model_accuracy(model):
    accuracy = model.accuracy(questions)
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, 
                                                                             percent(sum_corr), 
                                                                             percent(sum_incorr)))

# COMMAND ----------

def print_results(model):
    print('queen')
    for result in model.wv.most_similar("queen"):
        print(result)
    print()
    print('man')
    for result in model.wv.most_similar("man"):
        print(result)
    print()
    print('woman')    
    for result in model.wv.most_similar("woman"):
        print(result)
    print()
    print('frog')
    for result in model.wv.most_similar("frog"):
        print(result)
    print()
    print('awful')
    for result in model.wv.most_similar("awful"):
        print(result)
    print()
    print("breakfast cereal dinner lunch:")
    print(model.wv.doesnt_match("breakfast cereal dinner lunch".split()))
    print("captain onion starship alien:")
    print(model.wv.doesnt_match("captain onion starship alien".split()))
    print("father mother son daughter film:")
    print(model.wv.doesnt_match("father mother son daughter film".split()))
    print("france england germany berlin:")
    print(model.wv.doesnt_match("france england germany berlin".split()))
    print("woman", "girl")
    print(model.wv.similarity("woman", "girl"))
    print("woman", "man")
    print(model.wv.similarity("woman", "man"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### TED Model

# COMMAND ----------

#download the data

#url = "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip"
#urllib.request.urlretrieve(url, filename="ted_en-20160408.zip")

# extract subtitles
with zipfile.ZipFile('../datasets/Word2vec/ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))

# COMMAND ----------

# remove parenthesis 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    
# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

# COMMAND ----------

model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)

# COMMAND ----------

print_results(model_ted)

# COMMAND ----------

del model_ted

# COMMAND ----------

# MAGIC %md
# MAGIC ### Google News Model

# COMMAND ----------

googlenews = os.path.join('../datasets/Word2vec/GoogleNews-vectors-negative300.bin')

# COMMAND ----------

model_googlenews = gensim.models.KeyedVectors.load_word2vec_format(googlenews, binary=True)

# COMMAND ----------

w2v_model_accuracy(model_googlenews)

# COMMAND ----------

print_results(model_googlenews)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Media Cloud Model

# COMMAND ----------

#mediacloud = os.path.join(path_io_files, 'MediaCloud_w2v')
mediacloud = os.path.join(path_io_files, 'MediaCloud_w2v_trigrams')

# COMMAND ----------

model_mediacloud = gensim.models.Word2Vec.load(mediacloud)

# COMMAND ----------

model_mediacloud.most_similar('fgv')

# COMMAND ----------

def build_neighbors(word, model, nviz=15):
    g = nx.Graph()
    g.add_node(word, {'color':'blue'})
    viz1 = model.most_similar(word, topn=nviz)
    g.add_weighted_edges_from([(word, v, w) for v,w in viz1 if w> 0.5] )
    for v in viz1:
        g.add_weighted_edges_from([(v[0], v2, w2) for v2,w2 in model.most_similar(v[0])])
    return g

# COMMAND ----------

word = 'andré_braz'
g = build_neighbors(word, model_mediacloud)
cols = ['r']*len(g.nodes()); cols[g.nodes().index(word)]='b'
pos = nx.spring_layout(g, iterations=100)
nx.draw_networkx(g,pos=pos, node_color=cols, node_size=1000, alpha=0.5, font_size=16)
#nx.draw_networkx_labels(g, pos,dict(zip(g.nodes(),g.nodes())))

# COMMAND ----------

print_results(model_mediacloud)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wikipedia Model
# MAGIC 
# MAGIC (You'll need at least 36GB RAM to process this file)  

# COMMAND ----------

#http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
# Download the raw xml file from wikimedia
# https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

wikipedia = os.path.join(path_io_files,'enwiki-latest-pages-articles.xml.bz2')

# Use this tool to open the wikimedia dump
wiki = WikiCorpus(wikipedia, lemmatize=False, dictionary={})

# COMMAND ----------

# Create a new file and save the processed dump
with open(os.path.join(path_io_files,'wikimedia_processed_w2v'), 'w') as f:
    for text in wiki.get_texts():
        text = [token.decode('utf8') for token in text]
        f.write(' '.join(text) + "\n")

# COMMAND ----------

#Create a model based in the processed dump 
with open(os.path.join(path_io_files,'wikimedia_processed_w2v'), 'r') as f:
    model_wikipedia = Word2Vec(LineSentence(f), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

# COMMAND ----------

# Trim unneeded model memory = use (much) less RAM
model_wikipedia.init_sims(replace=True)
# Save as a model
model_wikipedia.save(os.path.join(path_io_files,'model_wikimedia_w2v'))

# COMMAND ----------

# Now you can load only the trimmed model and forget the other files
model_wikipedia = gensim.models.Word2Vec.load(os.path.join(path_io_files,'model_wikimedia_w2v'))

# COMMAND ----------

model_wikipedia.wv.vocab["tee"].count

# COMMAND ----------

w2v_model_accuracy(model_wikipedia)

# COMMAND ----------

print_results(model_wikipedia)
