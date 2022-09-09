# Databricks notebook source
import os
import pathlib

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### In this notebook, we will write a series of commands using Apache Spark, including:
# MAGIC 
# MAGIC + Configuring your notebook environment   
# MAGIC + Loading and exploring data  
# MAGIC + Visualizing data  
# MAGIC 
# MAGIC ### A) Configure Apache Spark
# MAGIC 
# MAGIC First, we will need to perform a few configuration operations on the Apache Spark session to get optimal performance. These will include:  
# MAGIC 
# MAGIC 1. Specifying a database in which to work  
# MAGIC 2. Configuring the number of shuffle partitions to use  
# MAGIC 
# MAGIC For this dataset, the most appropriate number of partitions is eight.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Step 1: Specify the database
# MAGIC 
# MAGIC We will use the database dsacademy_{username}. This cell:
# MAGIC + Creates this database if it does not exist
# MAGIC + Sets the database for use in this Spark session
# MAGIC + Defines a path variable for the location of the Delta files to be used throughout the course

# COMMAND ----------

spark.catalog.currentDatabase()

# COMMAND ----------

username = "Renato"
dbutils.widgets.text("username", username)
spark.sql(f"CREATE DATABASE IF NOT EXISTS dsacademy_{username}")
spark.sql(f"USE dsacademy_{username}")

# COMMAND ----------

spark.catalog.currentDatabase()

# COMMAND ----------

# MAGIC %sql
# MAGIC show databases;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2: Configure the Number of Shuffle Partitions
# MAGIC 
# MAGIC Recall that for this dataset, the most appropriate number of partitions is eight.

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B) Reviewing data
# MAGIC 
# MAGIC #### Review health tracker data
# MAGIC 
# MAGIC One common use case for working with Delta Lake is to collect and process Internet of Things (IoT) Data. Here, we provide a mock IoT sensor dataset for demonstration purposes. The data simulates heart rate data measured by a health tracker device.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 4: Load the data
# MAGIC 
# MAGIC Load the data as a Spark DataFrame from the raw directory. This is done using the .format("json") option.  

# COMMAND ----------

path = pathlib.PurePath(path).parents[1] / "Data/healthtracker/"
path

# COMMAND ----------

file_path = os.path.join(path, "health_tracker_data_2020_1.json")
health_tracker_data_2020_1_df = (spark.read.format("json").load(file_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### C) Visualize data
# MAGIC 
# MAGIC #### Step 1: Display the data
# MAGIC 
# MAGIC Strictly speaking, this is not part of the ETL process, but displaying the data gives us a look at the data that we are working with. 
# MAGIC 
# MAGIC We note a few phenomena in the data:
# MAGIC 
# MAGIC + Sensor anomalies - Sensors cannot record negative heart rates, so any negative values in the data are anomalies.  
# MAGIC + Wake/Sleep cycle - We notice that users have a consistent wake/sleep cycle alternating between steady high and low heart rates.  
# MAGIC + Elevated activity - Some users have irregular periods of high activity.  

# COMMAND ----------

display(health_tracker_data_2020_1_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2: Configure the visualization
# MAGIC 
# MAGIC Note that we have used a Databricks visualization to visualize the sensor data over time. We have used the following plot options to configure the visualization: 
# MAGIC 
# MAGIC + Keys: time
# MAGIC + Series groupings: device_id
# MAGIC + Values: heartrate
# MAGIC + Aggregation: SUM
# MAGIC + Display Type: Bar Chart
# MAGIC 
# MAGIC Now that we have a better idea of the data we're working with, let's move on to create a Parquet-based table from this data. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### D) Create a Parquet Table
# MAGIC 
# MAGIC Now that we have used Databricks to preview the data, we'll work through the process of creating a Parquet table. This table will be used in the next lesson to show the ease of converting existing Parquet tables to Delta tables.
# MAGIC 
# MAGIC The development pattern used to create a Parquet table is similar to that used in creating a Delta table. There are a few issues that arise as part of the process, however. In particular, working with Parquet-based tables often requires table repairs to work with them.
# MAGIC 
# MAGIC In subsequent lessons, we'll see that creating a Delta table does not have the same issues.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 1: Remove files in the /dbacademy/DLRS/healthtracker/processed directory
# MAGIC 
# MAGIC Next, we remove the files in the /dbacademy/DLRS/healthtracker/processed directory. This step will make the notebook idempotent. In other words, it could be run more than once without throwing errors or introducing extra files.

# COMMAND ----------

dbutils.fs.rm(health_tracker + "processed", recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 2: Transform the data 
# MAGIC 
# MAGIC We will perform data engineering on the data with the following transformations:
# MAGIC 
# MAGIC + Use the from_unixtime Spark SQL function to transform the unix timestamp into a time string
# MAGIC + Cast the time column to type timestamp to replace the column time
# MAGIC + Cast the time column to type date to create the column dte
# MAGIC + Select the columns in the order in which we would like them to be written
# MAGIC 
# MAGIC As this is a process that we will perform on each dataset as it is loaded we compose a function to perform the necessary transformations. This function, process_health_tracker_data, can be reused each time.

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime
 
def process_health_tracker_data(dataframe):
  return (
    dataframe
    .withColumn("time", from_unixtime("time"))
    .withColumnRenamed("device_id", "p_device_id")
    .withColumn("time", col("time").cast("timestamp"))
    .withColumn("dte", col("time").cast("date"))
    .withColumn("p_device_id", col("p_device_id").cast("integer"))
    .select("dte", "time", "heartrate", "name", "p_device_id")
    )
  
processedDF = process_health_tracker_data(health_tracker_data_2020_1_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 3: Write the Files to the processed directory
# MAGIC 
# MAGIC Note that we are partitioning the data by device id.

# COMMAND ----------

(processedDF.write
 .mode("overwrite")
 .format("parquet")
 .partitionBy("p_device_id")
 .save(health_tracker + "processed"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 4: Register the table in the metastore
# MAGIC 
# MAGIC Next, use Spark SQL to register the table in the metastore. We specify the table format as parquet and we refer to the location where we wrote the parquet files.

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC DROP TABLE IF EXISTS health_tracker_processed;
# MAGIC 
# MAGIC CREATE TABLE health_tracker_processed                        
# MAGIC USING PARQUET                
# MAGIC LOCATION "/dbacademy/$username/DLRS/healthtracker/processed"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 5: Verify and repair the Parquet-based Data Lake table  
# MAGIC ##### Step 5a: Count the records in the health_tracker_processed table
# MAGIC 
# MAGIC Per best practice, we have created a partitioned table. However, if you create a partitioned table from existing data, Spark SQL does not automatically discover the partitions and register them in the Metastore. Note that the count does not return results.

# COMMAND ----------

health_tracker_processed = spark.read.table("health_tracker_processed")
health_tracker_processed.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 5b: Register the partitions
# MAGIC 
# MAGIC To register the partitions, run the following to generate the partitions.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC MSCK REPAIR TABLE health_tracker_processed

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 5c: Count the records in the health_tracker_processed table
# MAGIC 
# MAGIC Count the records in the health_tracker_processed table. With the table repaired and the partitions registered, we now have results. We expect there to be 3720 records: five device measurements, 24 hours a day for 31 days.

# COMMAND ----------

health_tracker_processed.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Table Fundamentals
# MAGIC 
# MAGIC Recall that a Delta table consists of three things:
# MAGIC 
# MAGIC + The data files kept in object storage (AWS S3, Azure Data Lake Storage)
# MAGIC + The Delta Transaction Log saved with the data files in object storage 
# MAGIC + A table registered in the Metastore. This step is optional
# MAGIC 
# MAGIC You can create a Delta table by either of the following methods:
# MAGIC 
# MAGIC + Convert parquet files using the Delta Lake API
# MAGIC + Write new files using the Spark DataFrame writer with .format("delta")
# MAGIC 
# MAGIC Either of these will automatically create the Transaction Log in the same top-level directory as the data files. Optionally, you can also register the table in the Metastore.
# MAGIC 
# MAGIC ### 6. Creating a Delta table
# MAGIC 
# MAGIC Creating a table is one of the most fundamental actions performed when working with Delta Lake. With Delta Lake, you create tables:
# MAGIC 
# MAGIC + When ingesting new files into a Delta Table for the first time
# MAGIC + By transforming an existing Parquet table to a Delta table 

# COMMAND ----------

# MAGIC %md 
# MAGIC Throughout this section, we'll be writing files to the root location of the Databricks File System (DBFS). In general, best practice is to write files to your cloud object storage. We use DBFS root here for demonstration purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 6.1: Describe the health_tracker_processed table
# MAGIC 
# MAGIC Before we convert the health_tracker_processed table, let's use the DESCRIBE DETAIL Spark SQL command to display the attributes of the table.  
# MAGIC Note that the table has format PARQUET.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE DETAIL health_tracker_processed

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


