# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperopt
# MAGIC 
# MAGIC Hyperopt is a Python library for "serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions".
# MAGIC 
# MAGIC In the machine learning workflow, hyperopt can be used to distribute/parallelize the hyperparameter optimization process with more advanced optimization strategies than are available in other libraries.
# MAGIC 
# MAGIC There are two ways to scale hyperopt with Apache Spark:
# MAGIC * Use single-machine hyperopt with a distributed training algorithm (e.g. MLlib)
# MAGIC * Use distributed hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class. 
# MAGIC 
# MAGIC We will use single-machine hyperopt with MLlib, and also see how to use hyperopt to distribute the hyperparameter tuning of single node models with Scikit-Learn. 
# MAGIC 
# MAGIC Unfortunately you can’t use hyperopt to distribute the hyperparameter optimization for distributed training algorithms at this time. However, you do still get the benefit of using more advanced hyperparameter search algorthims (random search, TPE, etc.) with Spark ML.
# MAGIC 
# MAGIC 
# MAGIC Resources:
# MAGIC 0. [Documentation](http://hyperopt.github.io/hyperopt/scaleout/spark/)
# MAGIC 0. [Hyperopt on Databricks](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/index.html)
# MAGIC 0. [Hyperparameter Tuning with MLflow, Apache Spark MLlib and Hyperopt](https://databricks.com/blog/2019/06/07/hyperparameter-tuning-with-mlflow-apache-spark-mllib-and-hyperopt.html)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use hyperopt to find the optimal parameters for an MLlib model using TPE

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by loading in our Airbnb Dataset.

# COMMAND ----------

import os

# COMMAND ----------

## Put your name here
username = "renato"

dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS dsacademy_embedded_wave3_{username}")
spark.sql(f"USE dsacademy_embedded_wave3_{username}")
spark.conf.set("spark.sql.shuffle.partitions", 40)

spark.sql("SET spark.databricks.delta.formatCheck.enabled = false")
spark.sql("SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true")

# COMMAND ----------

deltaPath = os.path.join("/", "tmp", username)    #If we were writing to the root folder and not to the DBFS
if not os.path.exists(deltaPath):
    os.mkdir(deltaPath)
    
print(deltaPath)

airbnbDF = spark.read.format("delta").load(deltaPath)

# COMMAND ----------

airbnbDF.display()

# COMMAND ----------

(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC We will then create our random forest pipeline and regression evaluator.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features", handleInvalid='skip')

rf = RandomForestRegressor(labelCol="price", maxBins=56, seed=42)

pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

# COMMAND ----------

regressionEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we get to the hyperopt-specific part of the workflow.
# MAGIC 
# MAGIC First, we define our **objective function**. The objective function has two primary requirements:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC 
# MAGIC In this case, we are specifying values of `max_depth` and `num_trees` and returning the RMSE as our loss metric.
# MAGIC 
# MAGIC We are reconstructing our pipeline for the `RandomForestRegressor` to use the specified hyperparameter values.

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow

def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    # create a grid with our hyperparameters
    grid = (ParamGridBuilder()
      .addGrid(rf.maxDepth, [max_depth])
      .addGrid(rf.numTrees, [num_trees])
      .build())

    # cross validate the set of hyperparameters
    cv = CrossValidator(estimator=pipeline, 
                        estimatorParamMaps=grid, 
                        evaluator=regressionEvaluator, 
                        numFolds=3)

    cvModel = cv.fit(trainDF)

    # get our average RMSE across all three folds
    rmse = cvModel.avgMetrics[0]

    return {"loss": rmse, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we define our search space. 
# MAGIC 
# MAGIC This is similar to the parameter grid in a grid search process. However, we are only specifying the range of values rather than the individual, specific values to be tested. It's up to hyperopt's optimization algorithm to choose the actual values.
# MAGIC 
# MAGIC See the [documentation](https://github.com/hyperopt/hyperopt/wiki/FMin) for helpful tips on defining your search space.

# COMMAND ----------

from hyperopt import hp

search_space = {
  "max_depth": hp.randint("max_depth", 2, 5),
  "num_trees": hp.randint("num_trees", 10, 100)
}

# COMMAND ----------

# MAGIC %md
# MAGIC `fmin()` generates new hyperparameter configurations to use for your `objective_function`. It will evaluate 4 models in total, using the information from the previous models to make a more informative decision for the the next hyperparameter to try. 
# MAGIC 
# MAGIC Hyperopt allows for parallel hyperparameter tuning using either random search or Tree of Parzen Estimators (TPE). Note that in the cell below, we are importing `tpe`. According to the [documentation](http://hyperopt.github.io/hyperopt/scaleout/spark/), TPE is an adaptive algorithm that 
# MAGIC 
# MAGIC > iteratively explores the hyperparameter space. Each new hyperparameter setting tested will be chosen based on previous results. 
# MAGIC 
# MAGIC Hence, `tpe.suggest` is a Bayesian method.

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

from hyperopt import fmin, tpe, STATUS_OK, Trials
import numpy as np

# Creating a parent run
with mlflow.start_run():
    num_evals = 4
    trials = Trials()
    best_hyperparam = fmin(fn=objective_function, 
                           space=search_space,
                           algo=tpe.suggest, 
                           max_evals=num_evals,
                           trials=trials,
                           #rstate=np.random.RandomState(42)
                           rstate=np.random.default_rng(42) #https://github.com/hyperopt/hyperopt/issues/838
                          )
  
    # get optimal hyperparameter values
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
  
    # change RF to use optimal hyperparameter values (this is a stateful method)
    rf.setMaxDepth(best_max_depth)
    rf.setNumTrees(best_num_trees)
  
    # train pipeline on entire training data - this will use the updated RF values
    pipelineModel = pipeline.fit(trainDF)
  
    # evaluate final model on test data
    predDF = pipelineModel.transform(testDF)
    rmse = regressionEvaluator.evaluate(predDF)
  
    # Log param and metric for the final model
    mlflow.log_param("max_depth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Using Hyperopt with Scikit-Learn](http://hyperopt.github.io/hyperopt/scaleout/spark/)  
# MAGIC Below, we give an example workflow which tunes a scikit-learn model using SparkTrials. 
# MAGIC This example was adapted from the scikit-learn doc example for sparse logistic regression

# COMMAND ----------

df = airbnbDF.toPandas()
df.dropna(inplace=True)
y = df["price"].values
X = df[['accommodates', 'bedrooms', 'beds', 'minimum_nights',
       'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'bedrooms_na', 'beds_na',
       'review_scores_rating_na', 'review_scores_accuracy_na',
       'review_scores_cleanliness_na', 'review_scores_checkin_na',
       'review_scores_communication_na', 'review_scores_location_na',
       'review_scores_value_na']].values

# COMMAND ----------

print(X.shape)
print(y.shape)

# COMMAND ----------

from sklearn.preprocessing import KBinsDiscretizer
discret = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', dtype=np.float32)
y = discret.fit_transform(y.reshape(-1, 1)).reshape(-1)
print(y.shape)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# First, set up the scikit-learn workflow, wrapped within a function.
def train(params):
    """
    This is our main training function which we pass to Hyperopt.
    It takes in hyperparameter settings, fits a model based on those settings,
    evaluates the model, and returns the loss.

    :param params: map specifying the hyperparameter settings to test
    :return: loss for the fitted model
    """
    # We will tune 2 hyperparameters:
    #  regularization and the penalty type (L1 vs L2).
    regParam = float(params['regParam'])
    penalty = params['penalty']

    # Turn up tolerance for faster convergence
    clf = LogisticRegression(C=1.0 / regParam,
                             multi_class='multinomial',
                             penalty=penalty, 
                             solver='saga', 
                             tol=0.1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return {'loss': -score, 'status': STATUS_OK}

# Next, define a search space for Hyperopt.
search_space = {
   'penalty': hp.choice('penalty', ['l1', 'l2']),
   'regParam': hp.loguniform('regParam', -10.0, 0),
}

# Select a search algorithm for Hyperopt to use.
algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

# COMMAND ----------

# MAGIC %md
# MAGIC ##### We can run Hyperopt locally (only on the driver machine) by calling `fmin` without an explicit `trials` argument.

# COMMAND ----------

best_hyperparameters = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=32)

print(best_hyperparameters)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### We can distribute tuning across our Spark cluster by calling `fmin` with a `SparkTrials` instance.

# COMMAND ----------

spark_trials = SparkTrials()
best_hyperparameters = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    trials=spark_trials,
    max_evals=32)

print(best_hyperparameters)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparing with Grid Search of Scikit Learn

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
import numpy as np

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

search_space = {'penalty': ['l1', 'l2', 'none'],
                'C': np.logspace(0.1,1,3)
               }

clf = LogisticRegression(max_iter=5000,
                         solver='saga')

clf_lr = GridSearchCV(clf, param_grid=search_space)
clf_lr.fit(X_train, y_train)
score = clf_lr.score(X_test, y_test)

# COMMAND ----------

print(clf_lr.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC Adapted and updated from 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
