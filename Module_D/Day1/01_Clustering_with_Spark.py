# Databricks notebook source
# MAGIC %md
# MAGIC ## [Clustering with Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html)  & [Clustering with Spark](https://spark.apache.org/docs/latest/ml-clustering.html)  
# MAGIC 
# MAGIC This notebook describes clustering algorithms in [Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods) and the [Spark MLlib](https://spark.apache.org/docs/latest/ml-clustering.html).  
# MAGIC 
# MAGIC ### Table of Contents
# MAGIC 
# MAGIC + K-means
# MAGIC + Input Columns
# MAGIC + Output Columns
# MAGIC + Latent Dirichlet allocation (LDA)
# MAGIC + Bisecting k-means
# MAGIC + Gaussian Mixture Model (GMM)
# MAGIC + Input Columns
# MAGIC + Output Columns
# MAGIC + Power Iteration Clustering (PIC)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Disabling MLFlow

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking file paths

# COMMAND ----------

!ls 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking home folder

# COMMAND ----------

import os
os.getcwd()

# Notes on reading files for Spark within Databricks
# https://docs.databricks.com/_static/notebooks/files-in-repos.html

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1 - Kmeans Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Examining Data

# COMMAND ----------

!head ./data/kmeans_data.txt

# COMMAND ----------

!head ./data/sample_kmeans_data.txt

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1.1 - [k-means](http://en.wikipedia.org/wiki/K-means_clustering) with Scikit-Learn
# MAGIC 
# MAGIC Kmeans is one of the most commonly used clustering algorithms that clusters the data points into a predefined number of clusters.
# MAGIC 
# MAGIC [Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html) implements many forms of clustering

# COMMAND ----------

!pip install -U -q scikit-learn

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# COMMAND ----------

X = np.loadtxt(f"{os.getcwd()}/data/kmeans_data.txt", dtype="float")
print(X)

# COMMAND ----------

kmeans = KMeans(n_clusters=2, 
                init='k-means++',
                random_state=0)
kmeans.fit(X)
kmeans.labels_

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1.1 - [Silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)  
# MAGIC 
# MAGIC Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of [-1, 1].
# MAGIC 
# MAGIC Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.

# COMMAND ----------

range_n_clusters = [2, 3, 4, 5]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 - [K-means with Spark](https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/k_means_example.py)  
# MAGIC 
# MAGIC The MLlib implementation includes a parallelized variant of the k-means++ method called kmeans||.
# MAGIC 
# MAGIC KMeans is implemented as an Estimator and generates a KMeansModel as the base model.
# MAGIC 
# MAGIC 
# MAGIC <h3 id="input-columns">Input Columns</h3>
# MAGIC 
# MAGIC <table class="table">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Param name</th>
# MAGIC       <th align="left">Type(s)</th>
# MAGIC       <th align="left">Default</th>
# MAGIC       <th align="left">Description</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td>featuresCol</td>
# MAGIC       <td>Vector</td>
# MAGIC       <td>"features"</td>
# MAGIC       <td>Feature vector</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC 
# MAGIC <h3 id="output-columns">Output Columns</h3>
# MAGIC 
# MAGIC <table class="table">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Param name</th>
# MAGIC       <th align="left">Type(s)</th>
# MAGIC       <th align="left">Default</th>
# MAGIC       <th align="left">Description</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td>predictionCol</td>
# MAGIC       <td>Int</td>
# MAGIC       <td>"prediction"</td>
# MAGIC       <td>Predicted cluster center</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Loads data.
dataset = spark.read.format("libsvm").load(f"file:{os.getcwd()}/data/sample_kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2.1 - [Silhouette analysis](https://spark.apache.org/docs/latest/ml-clustering.html#k-means) with Spark

# COMMAND ----------

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2 - [Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation)   
# MAGIC Latent Dirichlet Allocation is a generative probabilistic model for collections of discrete dataset such as text corpora.  
# MAGIC It is also a topic model that is used for discovering abstract topics from a collection of documents.  
# MAGIC 
# MAGIC The graphical model of LDA is a three-level generative model:
# MAGIC 
# MAGIC ![](https://scikit-learn.org/stable/_images/lda_model_graph.png)
# MAGIC 
# MAGIC Note on notations presented in the graphical model above, which can be found in Hoffman et al. (2013):
# MAGIC 
# MAGIC + The corpus is a collection of  documents.
# MAGIC + A document is a sequence of  words.
# MAGIC + There are  topics in the corpus.
# MAGIC + The boxes represent repeated sampling.
# MAGIC 
# MAGIC In the graphical model, each node is a random variable and has a role in the generative process. A shaded node indicates an observed variable and an unshaded node indicates a hidden (latent) variable. In this case, words in the corpus are the only data that we observe. The latent variables determine the random mixture of topics in the corpus and the distribution of words in the documents. The goal of LDA is to use the observed words to infer the hidden topic structure.
# MAGIC 
# MAGIC When modeling text corpora, the model assumes the following generative process for a corpus with  documents and  topics, with  corresponding to n_components in the API:

# COMMAND ----------

# MAGIC %md
# MAGIC Examining Data

# COMMAND ----------

!head ./data/sample_lda_data.txt

# COMMAND ----------

!head ./data/sample_lda_libsvm_data.txt

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 2.1 - [Latent Dirichlet allocation (LDA) with Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

# COMMAND ----------

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

# COMMAND ----------

X = np.loadtxt(f"{os.getcwd()}/data/sample_lda_data.txt", dtype="float")
print(X)

# COMMAND ----------

# X is a feature matrix of token counts, similar to what CountVectorizer would produce on text.

lda = LatentDirichletAllocation(n_components=10,
                                random_state=0)
lda.fit(X)

# get topics for some given samples:
df = pd.DataFrame(lda.transform(X))
df

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 - [Latent Dirichlet allocation with Spark (LDA)](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/lda_example.py)  
# MAGIC 
# MAGIC LDA is implemented as an Estimator that supports both EMLDAOptimizer and OnlineLDAOptimizer, and generates a LDAModel as the base model.  
# MAGIC Expert users may cast a LDAModel generated by EMLDAOptimizer to a DistributedLDAModel if needed.

# COMMAND ----------

from pyspark.ml.clustering import LDA

# Loads data.
dataset = spark.read.format("libsvm").load(f"file:{os.getcwd()}/data/sample_lda_libsvm_data.txt")

# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(dataset)

ll = model.logLikelihood(dataset)
lp = model.logPerplexity(dataset)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# COMMAND ----------

# Describe topics.
topics = model.describeTopics(5)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# COMMAND ----------

# Shows the result
transformed = model.transform(dataset)
#transformed.show(truncate=False)
transformed.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 - [Hierarchical Cluster](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
# MAGIC 
# MAGIC Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample. See the [Wikipedia page](https://en.wikipedia.org/wiki/Hierarchical_clustering) for more details.  
# MAGIC 
# MAGIC Bisecting k-means is a kind of [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) using a divisive (or “top-down”) approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.  
# MAGIC 
# MAGIC Bisecting K-means can often be much faster than regular K-means, but it will generally produce a different clustering.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 - [Bisecting k-means with Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans)

# COMMAND ----------

from sklearn.cluster import BisectingKMeans

# COMMAND ----------

X = np.loadtxt(f"{os.getcwd()}/data/kmeans_data.txt", dtype="float")
print(X)

# COMMAND ----------

bisect_means = BisectingKMeans(n_clusters=2, random_state=0).fit(X)
print(bisect_means.labels_)
print(bisect_means.cluster_centers_)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 - [Bisecting k-means with Spark](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/bisecting_k_means_example.py)
# MAGIC 
# MAGIC   
# MAGIC 
# MAGIC BisectingKMeans is implemented as an Estimator and generates a BisectingKMeansModel as the base model.  

# COMMAND ----------

from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Loads data.
dataset = spark.read.format("libsvm").load(f"file:{os.getcwd()}/data/sample_kmeans_data.txt")

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dataset)

# Make predictions
predictions = model.transform(dataset)
predictions.display()

# COMMAND ----------

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 - Gaussian Mixture Models  
# MAGIC 
# MAGIC A [Gaussian Mixture Model](http://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model) represents a composite distribution whereby points are drawn from one of k Gaussian sub-distributions, each with its own probability.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 - [Gaussian Mixture Model (GMM) with Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)
# MAGIC 
# MAGIC The GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models. It can also draw confidence ellipsoids for multivariate models, and compute the Bayesian Information Criterion to assess the number of clusters in the data. A GaussianMixture.fit method is provided that learns a Gaussian Mixture Model from train data.

# COMMAND ----------

from sklearn.mixture import GaussianMixture

# COMMAND ----------

X = np.loadtxt(f"{os.getcwd()}/data/kmeans_data.txt", dtype="float")
print(X)

# COMMAND ----------

gm = GaussianMixture(n_components=2, random_state=0).fit(X)

print(gm.means_)
print(gm.predict([[0, 0, 3], [12, 3, 21]]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 - [Gaussian Mixture Model (GMM) with Spark](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/gaussian_mixture_example.py)
# MAGIC 
# MAGIC   
# MAGIC The spark.ml implementation uses the [expectation-maximization](http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm to induce the maximum-likelihood model given a set of samples.  
# MAGIC 
# MAGIC GaussianMixture is implemented as an Estimator and generates a GaussianMixtureModel as the base model.  
# MAGIC 
# MAGIC <h3 id="input-columns-1">Input Columns</h3>
# MAGIC 
# MAGIC <table class="table">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Param name</th>
# MAGIC       <th align="left">Type(s)</th>
# MAGIC       <th align="left">Default</th>
# MAGIC       <th align="left">Description</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td>featuresCol</td>
# MAGIC       <td>Vector</td>
# MAGIC       <td>"features"</td>
# MAGIC       <td>Feature vector</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>
# MAGIC 
# MAGIC <h3 id="output-columns-1">Output Columns</h3>
# MAGIC 
# MAGIC <table class="table">
# MAGIC   <thead>
# MAGIC     <tr>
# MAGIC       <th align="left">Param name</th>
# MAGIC       <th align="left">Type(s)</th>
# MAGIC       <th align="left">Default</th>
# MAGIC       <th align="left">Description</th>
# MAGIC     </tr>
# MAGIC   </thead>
# MAGIC   <tbody>
# MAGIC     <tr>
# MAGIC       <td>predictionCol</td>
# MAGIC       <td>Int</td>
# MAGIC       <td>"prediction"</td>
# MAGIC       <td>Predicted cluster center</td>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td>probabilityCol</td>
# MAGIC       <td>Vector</td>
# MAGIC       <td>"probability"</td>
# MAGIC       <td>Probability of each cluster</td>
# MAGIC     </tr>
# MAGIC   </tbody>
# MAGIC </table>

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture

# loads data
dataset = spark.read.format("libsvm").load(f"file:{os.getcwd()}/data/sample_kmeans_data.txt")

gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(dataset)

print("Gaussians shown as a DataFrame: ")
model.gaussiansDF.display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 5 - [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering)  
# MAGIC 
# MAGIC In multivariate statistics, spectral clustering techniques make use of the spectrum (eigenvalues) of the similarity matrix of the data to perform dimensionality reduction before clustering in fewer dimensions. The similarity matrix is provided as an input and consists of a quantitative assessment of the relative similarity of each pair of points in the dataset.
# MAGIC 
# MAGIC In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex, or more generally when a measure of the center and spread of the cluster is not a suitable description of the complete cluster, such as when clusters are nested circles on the 2D plane.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 - [Spectral Clustering with Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

# COMMAND ----------

from sklearn.cluster import SpectralClustering

# COMMAND ----------

X = np.array([[0, 1, 1.0],
              [0, 2, 1.0],
              [1, 2, 1.0],
              [3, 4, 1.0],
              [4, 0, 0.1]])
X

# COMMAND ----------

clustering = SpectralClustering(n_clusters=2,
                                assign_labels='kmeans',
                                random_state=0).fit(X)


print(clustering)

print(clustering.labels_)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 - [Power Iteration Clustering (PIC) with Spark](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/power_iteration_clustering_example.py)
# MAGIC 
# MAGIC Power Iteration Clustering (PIC) is a scalable graph clustering algorithm developed by [Lin and Cohen](http://www.cs.cmu.edu/~frank/papers/icml2010-pic-final.pdf). From the abstract: PIC finds a very low-dimensional embedding of a dataset using truncated power iteration on a normalized pair-wise similarity matrix of the data.
# MAGIC 
# MAGIC spark.ml’s PowerIterationClustering implementation takes the following parameters:
# MAGIC 
# MAGIC + k: the number of clusters to create
# MAGIC + initMode: param for the initialization algorithm
# MAGIC + maxIter: param for maximum number of iterations
# MAGIC + srcCol: param for the name of the input column for source vertex IDs
# MAGIC + dstCol: name of the input column for destination vertex IDs
# MAGIC + weightCol: Param for weight column name

# COMMAND ----------

from pyspark.ml.clustering import PowerIterationClustering

df = spark.createDataFrame([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 2, 1.0),
    (3, 4, 1.0),
    (4, 0, 0.1)
], ["src", "dst", "weight"])

pic = PowerIterationClustering(k=2, 
                               maxIter=20, 
                               initMode="degree", 
                               weightCol="weight")

# Shows the cluster assignment
pic.assignClusters(df).display()
