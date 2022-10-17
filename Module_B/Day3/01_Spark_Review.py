# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Review
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Create a Spark DataFrame
# MAGIC  - Analyze the Spark UI
# MAGIC  - Cache data
# MAGIC  - Go between Pandas and Spark DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

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

# MAGIC %md
# MAGIC ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# MAGIC %md
# MAGIC Why were no Spark jobs kicked off above? Well, we didn't have to actually "touch" our data, so Spark didn't need to execute anything across the cluster.

# COMMAND ----------

#display(df.sample(.001))
df.sample(.001).limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Views
# MAGIC 
# MAGIC How can I access this in SQL?

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Count
# MAGIC 
# MAGIC Let's see how many records we have.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark UI
# MAGIC 
# MAGIC Open up the Spark UI - what are the shuffle read and shuffle write fields? The command below should give you a clue.

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cache
# MAGIC 
# MAGIC For repeated access, it will be much faster if we cache our data.  
# MAGIC Spark Cache and Persist are optimization techniques in DataFrame / Dataset for iterative and interactive Spark applications to improve the performance of Jobs. In this article, you will learn What is Spark cache() and persist(), how to use it in DataFrame, understanding the difference between Caching and Persistance and how to use these two with DataFrame, and Dataset using Scala examples.  
# MAGIC 
# MAGIC Though Spark provides computation 100 x times faster than traditional Map Reduce jobs, If you have not designed the jobs to reuse the repeating computations you will see degrade in performance when you are dealing with billions or trillions of data. Hence, we may need to look at the stages and use optimization techniques as one of the ways to improve performance.  
# MAGIC 
# MAGIC Using cache() and persist() methods, Spark provides an optimization mechanism to store the intermediate computation of a Spark DataFrame so they can be reused in subsequent actions.  
# MAGIC 
# MAGIC When you persist a dataset, each node stores its partitioned data in memory and reuses them in other actions on that dataset. And Spark’s persisted data on nodes are fault-tolerant meaning if any partition of a Dataset is lost, it will automatically be recomputed using the original transformations that created it.  

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-run Count
# MAGIC 
# MAGIC Wow! Look at how much faster it is now!

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examining available Datasets

# COMMAND ----------

# MAGIC %fs mounts

# COMMAND ----------

#%fs ls /databricks-datasets/
files = dbutils.fs.ls("/databricks-datasets") 
display(files)

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/iot-stream/data-device/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Debug Slow Query: Spark UI
# MAGIC 
# MAGIC Why is the query below slow? How can you speed it up?

# COMMAND ----------

DFjson = (spark.read.json("/databricks-datasets/iot-stream/data-device/"))

# COMMAND ----------

DFjson.count()

# COMMAND ----------

DFjson.cache().count()

# COMMAND ----------

DFjson.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect Data
# MAGIC 
# MAGIC When you pull data back to the driver  (e.g. call `.collect()`, `.toPandas()`,  etc), you'll need to be careful of how much data you're bringing back. Otherwise, you might get OOM exceptions!
# MAGIC 
# MAGIC A best practice is explicitly limit the number of records, unless you know your data set is small, before calling `.collect()` or `.toPandas()`.

# COMMAND ----------

df.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Some material adapted from Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
