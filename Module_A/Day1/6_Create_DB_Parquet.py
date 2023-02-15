# Databricks notebook source
# MAGIC %md
# MAGIC ### Create database from parquet files in DBFS

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking and changing the current Database

# COMMAND ----------

spark.catalog.currentDatabase()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC You will now create databases in your workspace.  
# MAGIC To name it correctly, **substitute "renato" by your name in the `username` variable**.

# COMMAND ----------

## Put your name here

username = "renato"

dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS dsacademy_embedded_wave3_{username}")
spark.sql(f"USE dsacademy_embedded_wave3_{username}")

spark.sql("SET spark.databricks.delta.formatCheck.enabled = false")
spark.sql("SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking database creation  

# COMMAND ----------

spark.sql('SHOW DATABASES').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking existing tables

# COMMAND ----------

spark.sql('SHOW TABLES').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking example parquet files  

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

# MAGIC %md #####Checking inside folder

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/

# COMMAND ----------

# MAGIC %md #####Checking inside subfolder

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/amazon/

# COMMAND ----------

# MAGIC %md #####Creating a SPARK Dataframe from a Parquet file:

# COMMAND ----------

path = 'dbfs:/databricks-datasets/amazon/data20K/' # <-- change this to your S3 bucket of my use case
df = spark.read.format('parquet').options(header=True,inferSchema=True).load(path)
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create tables from Parquet files in DBFS  

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS data20k USING parquet OPTIONS (path "dbfs:/databricks-datasets/amazon/data20K/");
# MAGIC CREATE TABLE IF NOT EXISTS data40k USING parquet OPTIONS (path "dbfs:/databricks-datasets/amazon/test4K/");

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking tables creation  
# MAGIC 
# MAGIC You can also use the **data** tab of the workspace UI to confirm your tables were created.

# COMMAND ----------

spark.sql('SHOW TABLES').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute SQL to explore the newly created datasets  
# MAGIC Run SQL queries on the `data20k`, and `data40k` tables to answer the following questions.   
# MAGIC - How many brands are available for purchasing at Amazon?  
# MAGIC - What is the average rating for brand Greenies?  
# MAGIC - Which brand has the highest average price?  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1: What brands are available for purchasing at Amazon?  
# MAGIC 
# MAGIC We can inspect the structure of the **`data40k`** dataset.

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE data40k;

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM data40k LIMIT 100;

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- ANSWER
# MAGIC SELECT COUNT(DISTINCT(brand)) FROM data40k;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2: What is the average rating for brand Greenies?  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC SELECT AVG(rating)
# MAGIC FROM data40k
# MAGIC WHERE brand = 'Greenies';

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3: Which brand has the highest average price?

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC SELECT brand, AVG(price) as avg_price
# MAGIC FROM data40k
# MAGIC GROUP BY brand
# MAGIC ORDER BY avg_price
# MAGIC DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC You can always use the _sqldf variable that stires the result of the last query in a Spark dataframe.

# COMMAND ----------

# _sqldf.display()
_sqldf.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Now it's you turn!
# MAGIC 
# MAGIC Run SQL queries on the data40k tables to answer the following questions.
# MAGIC 
# MAGIC + Which brand has more reviews with the word "love"?
# MAGIC + Is it reasonable to expect that more expensive products receive more ratings? And more positive ratings? Answer this using visualizations. 
# MAGIC + Which brand has the most helpful reviews?

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
