# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## From Pandas to PySpark 
# MAGIC 
# MAGIC Some parts of this material were taken from [Databricks Academy](https://customer-academy.databricks.com/) and from this [Source](https://medium.com/@bhanusree.balisetty/from-pandas-to-pyspark-e7188c8276e).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### We love Pandas!  
# MAGIC 
# MAGIC Learning programming with Pandas is like getting started with the “Hello World” program in the world of data science.  
# MAGIC Pandas is a widely used, intuitive, easy to learn Python library. It deals with dataframes which store data in tabular format with rows and columns (spreadsheet format). Pandas loads all the data into the memory of the machine (Single Node) for faster execution.  
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/1280px-Grosser_Panda.JPG" width="512" height="384" />
# MAGIC 
# MAGIC #### Why Pyspark then?
# MAGIC 
# MAGIC While Pandas stays one of the widely used libraries in dealing with tabular data, especially in Data Science, it does not fully support **parallelization**.  
# MAGIC Pyspark is a Python API for Spark. It has been released to support the collaboration between Python and Spark environments.
# MAGIC 
# MAGIC Pyspark with its *cluster computing* processes the data in a distributed manner by running the code on multiple nodes, leading to decreased execution times. With data being created exponentially every day, Data Scientists now have huge datasets to deal with, which is where distributed computing comes in. But what is Spark?

# COMMAND ----------

# MAGIC %md #### Apache Spark
# MAGIC 
# MAGIC A single computer usually has the memory and computational power to perform calculations on data sets up to the size of a few gigabytes or less. Data sets larger than that either can't fit into the memory of a single computer or take an unacceptably long time for a single computer to process. For these types of "big data" use cases, we need a system that can split a large data set into smaller subsets &mdash; often referred to as **partitions** &mdash; and then distribute the processing of these data partitions across a number of computers.
# MAGIC 
# MAGIC [Apache Spark](https://spark.apache.org/) is an open-source data processing engine that manages distributed processing of large data sets. The creaters of Spark also founded Databricks. Databricks, in general, uses Apache Spark as the computation engine for the platform.
# MAGIC 
# MAGIC Let's say that we have a large data set and we want to calculate various statistics for some of its numeric columns. With Apache Spark, our program only needs to specify the data set to read and the statistics that we want calculated. We can then run the program on a set of computers that have been configured to serve as an Apache Spark **cluster**. When we run it, Spark automatically
# MAGIC 
# MAGIC * determines how to divide the data set into partitions,
# MAGIC * assigns those partitions to the various computers of the cluster with instructions for calculating per-partition statistics, and
# MAGIC * finally collects those per-partitions statistics and calculates the final results we requested.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/sparkcluster.png" style="width:600px;height:250px;">
# MAGIC 
# MAGIC If you want to move from a single node to multiple nodes and adapt to distributed cluster computing, this notebook will help in converting Pandas code to Pyspark code.  
# MAGIC It presents some of the commonly used Pandas Dataframe transformations and some miscellaneous operations along with the corresponding Pyspark syntax.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by importing the necessary packages:  

# COMMAND ----------

import os
import pandas as pd
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC In Databricks notebooks, the SparkSession is created for you, stored in a variable called spark. That is why the line below is commented:

# COMMAND ----------

# spark = SparkSession.builder.appName('spark_session').getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![](https://miro.medium.com/max/1280/1*aJSwrFLDlDbf9axjJ7gXHw.jpeg)
# MAGIC 
# MAGIC The SparkSession class is the single entry point to all functionality in Spark using the DataFrame API.  
# MAGIC It provides a way to interact with various spark’s functionality in order to programmatically create the PySpark RDD (Resilient Distributed Dataset).   
# MAGIC Instead of having a spark context, hive context, SQL context, now all of it is encapsulated in a Spark session.
# MAGIC More information [here](https://sparkbyexamples.com/pyspark/pyspark-what-is-sparksession/).

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md
# MAGIC #### In the next sections, we will compare Pandas and PySpark regarding the following tasks:  
# MAGIC 
# MAGIC 1. Creating Dataframes
# MAGIC 2. Creating new Columns
# MAGIC 3. Updating existing Column data
# MAGIC 4. Select and Filtering data
# MAGIC 5. Column Type Transformations
# MAGIC 6. Rename, Drop Columns
# MAGIC 7. Melt Dataframes
# MAGIC 8. Add Interval to a Timestamp Column (Timedelta)
# MAGIC 9. Additional Syntax

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 - Creating Dataframes

# COMMAND ----------

# MAGIC %md 
# MAGIC Let us take a look at how we create dataframes from scratch in Pandas as compared to Pyspark. 

# COMMAND ----------

# PANDAS
df1 = [['A1', 'B1', 2, '21-12-2021 10:30'], 
       ['A2', 'B2', 4, '21-12-2021 10:40'], 
       ['A3', 'B3', 5, '21-12-2021 11:00']] 

df1 = pd.DataFrame(df1, columns = ['A', 'B', 'Value', 'Date_Column'])
df1.head()

# COMMAND ----------

# PYSPARK
df2 = spark.createDataFrame([('A1', 'B1', 2, '21-12-2021 10:30'),
                            ('A2', 'B2', 4, '21-12-2021 10:40'),
                            ('A3', 'B3', 5, '21-12-2021 11:00')],
                            ['A', 'B', 'Value', 'Date_Column'])
df2.show()

# COMMAND ----------

df2.show(n=3, truncate=False, vertical=True)  #print top 3 rows vertically

# COMMAND ----------

df2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2 - Creating New Columns

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we compare how we create new columns. 
# MAGIC If you are coming from Pandas, the Pyspark syntax might be less intuitive at first.   
# MAGIC 
# MAGIC In Pandas, you can create a new column by simply assigning a value to it. For example:

# COMMAND ----------

# PANDAS - New column with constant values
df1['C'] = 'New Constant'
df1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In Spark, the `withColumn()` method works differently. It takes two arguments: the name of the new column and a Column object that defines the values to be placed in the new column. The Column object is created using functions from the `pyspark.sql.functions` module. For example:

# COMMAND ----------

# PYSPARK - New column with nonstant values
df2 = df2.withColumn("C", F.lit('New Constant'))
df2.show()

# COMMAND ----------

# PANDAS - New Column using existing columns
df1['C'] = df1['A'] + df1['B']
df1.head()

# COMMAND ----------

# PYSPARK - New Column using existing columns
df2 = df2.withColumn("C", F.concat("A", "B"))
df2.show()



# COMMAND ----------

# MAGIC %md
# MAGIC #### NOTE
# MAGIC - ``lit()`` – used to create constant columns
# MAGIC - ``concat()`` – concatenate columns of dataframe
# MAGIC - ``withColumn()`` – creates a new column

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 - Updating Existing Column Data
# MAGIC In Pandas, you can update the values in a column by simply assigning new values to it using the indexing operator. 

# COMMAND ----------

# PANDAS - Update Column data
df1['Value'] = df1['Value']**2
df1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark, you can update the values in a column using the withColumn method and a function from the pyspark.sql.functions module.

# COMMAND ----------

# PYSPARK - Update Column data
df2 = df2.withColumn("Value", F.col("Value")**2)
df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4 - Selecting and Filtering Data

# COMMAND ----------

# MAGIC %md
# MAGIC In Pandas, you can select a single column or multiple columns from a dataframe by using the indexing operator and passing a list of column names.

# COMMAND ----------

# PANDAS - Selecting Columns
new_df1 = df1[['B', 'C']]
new_df1.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC In Spark, you can select a single column or multiple columns from a dataframe by using the `select` method and passing either the column names or a list of column names.

# COMMAND ----------

# PYSPARK - Selecting Columns
new_df2 = df2.select("B", "C")
new_df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC In both Pandas and Spark, you can filter rows from a dataframe based on conditions.   
# MAGIC In Pandas, you can use the indexing operator and boolean masks to filter rows. 

# COMMAND ----------

# PANDAS - Filtering rows based on condition
new_df1 = df1[df1['Value']<5]
new_df1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark, you can use the `filter` method to filter rows based on a condition by passing the condition as a Column object.

# COMMAND ----------

# PYSPARK - Filtering rows based on condition
new_df2 = df2.filter(df2.Value<5)
new_df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5 - Column Type Transformations

# COMMAND ----------

# MAGIC %md
# MAGIC In both Pandas and Spark, you can convert a column from string to datetime/timestamp format.  
# MAGIC In Pandas, you can use the `to_datetime` function to convert a column from string to datetime format by passing the column name as an argument. For example:

# COMMAND ----------

# PANDAS - Convert Column from String to DateTime format
df1['Date_Column'] =  pd.to_datetime(df1['Date_Column'], format='%d-%m-%Y %H:%M')
df1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In Spark, you can use the to_timestamp function from the pyspark.sql.functions module to convert a column from string to timestamp format. 

# COMMAND ----------

# PYSPARK - Convert Column from String to Timestamp format
df2 = df2.withColumn("Date_Column", F.to_timestamp("Date_Column", "dd-MM-yyyy hh:mm"))
df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6 - Rename, Drop Columns
# MAGIC 
# MAGIC In both Pandas and Spark, you can rename and drop columns in a dataframe. However, there are some differences in the way this is done between the two libraries.
# MAGIC 
# MAGIC In Pandas, you can rename columns using the `rename` method and passing a dictionary that maps the old column names to the new column names.

# COMMAND ----------

# PANDAS - Rename Columns
df1 = df1.rename(columns={'A': 'Col_A', 'B': 'Col_B'})
df1.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC In Spark, you can rename columns using the `withColumnRenamed` method and passing the old and new column names. 

# COMMAND ----------

# PYSPARK - Rename Columns
df2 = df2.withColumnRenamed("A", "Col_A").withColumnRenamed("B", "Col_B")
df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC You can also drop columns in Pandas using the `drop` method and passing a list of column names to be dropped.

# COMMAND ----------

# PANDAS - Drop Columns
df1 = df1.drop(['Col_A', 'Col_B'], axis=1)
df1.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC In Spark you drop columns by using the `drop` method and passing a list of column names to be dropped. 

# COMMAND ----------

# PYSPARK - Drop Columns
df2 = df2.drop('A', 'B')
df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 - Melt Dataframes
# MAGIC Melting a dataframe refers to the process of transforming a dataframe from a wide format to a long format. In a wide format dataframe, each row represents a single observation, and each column represents a variable. In a long format dataframe, each row represents a single observation and a single variable.  
# MAGIC 
# MAGIC To melt a dataframe, you need to specify the columns that contain the variables to be melted and the columns that contain the values for those variables. The result will be a new dataframe where the specified columns are "unpivoted" into two new columns: one for the variable names and one for the values.

# COMMAND ----------

# MAGIC %md
# MAGIC In Pandas, you can use the `melt` function to melt a dataframe. The melt function takes a number of arguments, including the `id_vars` and `value_vars` arguments, which specify the columns that should be kept as identifiers and the columns that should be melted, respectively. 

# COMMAND ----------

# PANDAS
df3 = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                    'B': {0: 1, 1: 3, 2: 5},
                    'C': {0: 2, 1: 4, 2: 6}})

pd.melt(df3, id_vars=['A'], value_vars=['B', 'C'])

# COMMAND ----------

# MAGIC %md 
# MAGIC In Spark, there is no built-in function for melting dataframes. However, you can use the `withColumn` method and functions to achieve the same result.

# COMMAND ----------

# PYSPARK custom melt function
def melt(df, id_vars, value_vars, var_name="Variable", value_name="Value"):
    _vars_and_vals = F.array(*(F.struct(F.lit(c).alias(var_name),
                                        F.col(c).alias(value_name)) for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals",
                         F.explode(_vars_and_vals))
    cols = id_vars + [F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


df4 = spark.createDataFrame([('a', 1, 2), ('b', 3, 4), ('c', 5, 6)], ['A', 'B', 'C'])

melt(df4, ['A'], ['B', 'C']).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8 - Add Interval to a Timestamp Column (Timedelta)
# MAGIC 
# MAGIC A timedelta is a duration that represents a difference between two dates or times. 
# MAGIC 
# MAGIC You might want to add a timedelta to a timestamp column in a DataFrame for several reasons. For example, you might want to shift all the timestamps forward by a certain duration to correct for an error in the original data, or to change the time zone of the timestamps. Another reason to add a timedelta is to create a new column with modified timestamps, such as events that are scheduled to occur at a later time than the events in the original column. Additionally, you might want to group the data by week, month, or year and compute summary statistics for each group by adding a timedelta to the timestamp column and shifting all the timestamps so that they fall within the same week, month, or year.

# COMMAND ----------

# MAGIC %md 
# MAGIC In both Pandas and Spark, you can add a timedelta or time interval to a timestamp column by using the `+` operator. 
# MAGIC 
# MAGIC In Pandas, a timedelta is represented by a `Timedelta` object, which can be created using the `timedelta` function. Alternatively, you can use the `to_timedelta` function to convert a column of data to a `Timedelta` type.

# COMMAND ----------

# PANDAS - Add 'Interval' to 'Start_Time'
df5 = pd.DataFrame([['2021-01-10 10:10:00', '00:05'],
                    ['2021-12-10, 05:30:00', '00:15'],
                    ['2021-11-10 11:40:00', '00:20']], 
                   columns = ['Start_Time','Interval'])

df5['Start_Time'] = pd.to_datetime(df5['Start_Time'])
df5['End_Time'] = df5['Start_Time'] + pd.to_timedelta(pd.to_datetime(df5['Interval']).dt.strftime('%H:%M:%S'))
df5.head()

# COMMAND ----------

# MAGIC %md
# MAGIC In PySpark, a timedelta is represented by a usual column in timestamp format. Hence, you can use the `to_timestamp` function to convert a column of data to a timestamp type as previously seen.

# COMMAND ----------

# PYSPARK - Add 'Interval' to 'Start_Time'
df6 = spark.createDataFrame([['2021-01-10 10:10:00', '00:05'], 
                            ['2021-12-10 05:30:00', '00:15'], 
                            ['2021-11-10 11:40:00', '00:20']], 
                            ['Start_Time', 'Interval'])

df6 = df6.withColumn("Start_Time", F.to_timestamp("Start_Time", "yyyy-MM-dd hh:mm:ss"))
df6 = df6.withColumn("End_Time", (F.unix_timestamp("Start_Time") + F.unix_timestamp("Interval", "HH:mm")).cast('timestamp'))
df6.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9 - Converting from Spark to Pandas and vice versa

# COMMAND ----------

# MAGIC %md
# MAGIC We have two dataframes, one being a Pandas dataframe, the other being a Pyspark dataframe. 

# COMMAND ----------

print(type(df5))
print(type(df6))

# COMMAND ----------

# MAGIC %md
# MAGIC We want to convert from Pandas to Spark and from Spark to Pandas. 
# MAGIC 
# MAGIC To convert from Pandas to Spark, we use `createDataFrame()`.

# COMMAND ----------

sparkDF = spark.createDataFrame(df5) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC On the other hand, to convert from Spark to Pandas, we use the `toPandas()` method.

# COMMAND ----------

pandasDF = df6.toPandas()
pandasDF.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 10 - Additional Syntax

# COMMAND ----------

# PANDAS df
df7 = pd.DataFrame({'A': {0: 'a', 1: 'a', 2: 'c'},
                    'B': {0: 1, 1: 1, 2: 5},
                    'C': {0: 2, 1: 4, 2: 6}})
df7

# COMMAND ----------

# PANDAS - Shape of dataframe
print(df7.shape)

# COMMAND ----------

# PANDAS - Distinct values of a column
df7['A'].unique()

# COMMAND ----------

# PANDAS - Group by columns - Calculate aggregate functions
df7.groupby(['A', 'B']).sum()

# COMMAND ----------

# PYSPARK df
df8 = spark.createDataFrame([('a', 1, 2), ('a', 1, 4), ('c', 5, 6)],
                            ['A', 'B', 'C'])

# COMMAND ----------

# PYSPARK - Shape of dataframe
print((df8.count(), len(df8.columns)))

# COMMAND ----------

# PYSPARK - Distinct values of a column
df8.select('A').distinct().show()

# COMMAND ----------

# PYSPARK - Group by columns - calculate aggregate functions
df8.groupBy("A", "B").agg(F.sum("C")).show()
