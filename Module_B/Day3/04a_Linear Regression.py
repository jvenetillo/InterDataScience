# Databricks notebook source
# MAGIC %md
# MAGIC # Regression: Predicting Rental Price
# MAGIC 
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use the SparkML API to build a linear regression model
# MAGIC  - Identify the differences between estimators and transformers

# COMMAND ----------

import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting the default database and user name  
# MAGIC ##### Substitute "renato" by your name in the `username` variable.

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

airbnbDF.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)
# MAGIC 
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)
print(trainDF.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set.

# COMMAND ----------

trainRepartitionDF, testRepartitionDF = (airbnbDF
                                         .repartition(24)
                                         .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC 
# MAGIC We are going to build a very simple model predicting `price` just given the number of `bedrooms`.
# MAGIC 
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

display(trainDF.select("price", "bedrooms"))

# COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC There do appear some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models :).
# MAGIC 
# MAGIC We will use `LinearRegression` to build our first model [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression).
# MAGIC 
# MAGIC The cell below will fail because the Linear Regression estimator expects a vector of values as input. We will fix that with VectorAssembler below.  
# MAGIC https://stackoverflow.com/questions/61056160/illegalargumentexception-column-must-be-of-type-structtypetinyint-sizeint-in

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="bedrooms", labelCol="price")

# Uncomment when running
lrModel = lr.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC 
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (`.fit()`) expected a column of Vector type as input.
# MAGIC 
# MAGIC We can easily get the values from the `bedrooms` column into a single vector using `VectorAssembler` [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.
# MAGIC 
# MAGIC You can see an example of how to use VectorAssembler on the [ML Programming Guide](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler).

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vecTrainDF = vecAssembler.transform(trainDF)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the model

# COMMAND ----------

m = lrModel.coefficients[0]
b = lrModel.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

vecTestDF = vecAssembler.transform(testDF)

predDF = lrModel.transform(vecTestDF)

predDF.select("bedrooms", "features", "price", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Model
# MAGIC 
# MAGIC Let's see how our linear regression model with just one variable does.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE is {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### It's still not that great. Let's see how we can further decrease it in the next notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC Code modified and enhanced from 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
