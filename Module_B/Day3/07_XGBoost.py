# Databricks notebook source
#!pip install sparkxgb

# COMMAND ----------

# MAGIC %md
# MAGIC # XGBoost
# MAGIC 
# MAGIC #### Using the example at: [This repo](https://github.com/sllynn/spark-xgboost/blob/master/examples/spark-xgboost_adultdataset.ipynb)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing modules and disabling MLflow  

# COMMAND ----------

from sparkxgb import XGBoostClassifier, XGBoostRegressor
from pprint import PrettyPrinter

from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
pp = PrettyPrinter()

# COMMAND ----------

col_names = [
  "age", "workclass", "fnlwgt",
  "education", "education-num",
  "marital-status", "occupation",
  "relationship", "race", "sex",
  "capital-gain", "capital-loss",
  "hours-per-week", "native-country",
  "label"
]

train_sdf, test_sdf = (
  spark.read.csv(
    path="/databricks-datasets/adult/adult.data",
    inferSchema=True  
  )
  .toDF(*col_names)
  .repartition(200)
  .randomSplit([0.8, 0.2])
)

# COMMAND ----------

string_columns = [fld.name for fld in train_sdf.schema.fields if isinstance(fld.dataType, StringType)]
string_col_replacements = [fld + "_ix" for fld in string_columns]
string_column_map=list(zip(string_columns, string_col_replacements))
target = string_col_replacements[-1]
predictors = [fld.name for fld in train_sdf.schema.fields if not isinstance(fld.dataType, StringType)] + string_col_replacements[:-1]
pp.pprint(
  dict(
    string_column_map=string_column_map,
    target_variable=target,
    predictor_variables=predictors
  )
)

# COMMAND ----------

si = [StringIndexer(inputCol=fld[0], outputCol=fld[1]) for fld in string_column_map]
va = VectorAssembler(inputCols=predictors, outputCol="features")
pipeline = Pipeline(stages=[*si, va])
fitted_pipeline = pipeline.fit(train_sdf.union(test_sdf))

# COMMAND ----------

train_sdf_prepared = fitted_pipeline.transform(train_sdf)
train_sdf_prepared.cache()
train_sdf_prepared.count()

# COMMAND ----------

test_sdf_prepared = fitted_pipeline.transform(test_sdf)
test_sdf_prepared.cache()
test_sdf_prepared.count()

# COMMAND ----------

xgbParams = dict(
  eta=0.1,
  maxDepth=2,
  missing=0.0,
  objective="binary:logistic",
  numRound=5,
  numWorkers=2
)

xgb = (
  XGBoostClassifier(**xgbParams)
  .setFeaturesCol("features")
  .setLabelCol("label_ix")
)

bce = BinaryClassificationEvaluator(
  rawPredictionCol="rawPrediction",
  labelCol="label_ix"
)

# COMMAND ----------

param_grid = (
  ParamGridBuilder()
  .addGrid(xgb.eta, [1e-1, 1e-2, 1e-3])
  .addGrid(xgb.maxDepth, [2, 4, 8])
  .build()
)

cv = CrossValidator(
  estimator=xgb,
  estimatorParamMaps=param_grid,
  evaluator=bce,#mce,
  numFolds=5
)

# COMMAND ----------

import mlflow
import mlflow.spark

spark_model_name = "best_model_spark"

with mlflow.start_run():
  model = cv.fit(train_sdf_prepared)
  best_params = dict(
    eta_best=model.bestModel.getEta(),
    maxDepth_best=model.bestModel.getMaxDepth()
  )
  mlflow.log_params(best_params)
  
  mlflow.spark.log_model(fitted_pipeline, "featuriser")
  mlflow.spark.log_model(model, spark_model_name)

  metrics = dict(
    roc_test=bce.evaluate(model.transform(test_sdf_prepared))
  )
  mlflow.log_metrics(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative Gradient Boosted Approaches
# MAGIC 
# MAGIC There are lots of other gradient boosted approaches, such as [CatBoost](https://catboost.ai/), [LightGBM](https://github.com/microsoft/LightGBM), vanilla gradient boosted trees in [SparkML](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassifier)/[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), etc. Each of these has their respective [pros and cons](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db) that you can read more about.

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
