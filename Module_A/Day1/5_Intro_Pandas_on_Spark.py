# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Short introduction to the Pandas API on Spark
# MAGIC 
# MAGIC 
# MAGIC **Pandas** is a great tool to analyze small datasets on a single machine. As we have discussed in a previous notebook, once the need for bigger datasets arises, users often choose **PySpark**.  
# MAGIC However, converting code from Pandas to PySpark is not easy as PySpark APIs are considerably different from pandas APIs.  
# MAGIC 
# MAGIC 
# MAGIC The **Pandas API on Spark**, also called **Pandas-on-Spark**, was created to make this learning curve significantly easier by providing pandas-like APIs on the top of PySpark. 
# MAGIC With the Pandas API on Spark, users can take advantage of the benefits of PySpark with minimal efforts, and thus get to value much faster.
# MAGIC 
# MAGIC 
# MAGIC + [Pandas-on-Spark User Guide](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html)
# MAGIC 
# MAGIC This is a short introduction to Pandas-on-Spark, geared mainly at new users.  
# MAGIC This notebook shows you some key differences between Pandas and Pandas-on-Spark.  
# MAGIC 
# MAGIC Customarily, we import Pandas-on-Spark as follows:

# COMMAND ----------

import pyspark.pandas as ps

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object Creation

# COMMAND ----------

# MAGIC %md
# MAGIC You can create a Pandas-on-Spark series by passing a list of values, letting Pandas-on-Spark create a default integer index:

# COMMAND ----------

s = ps.Series([1, 3, 5, np.nan, 6, 8])
s

# COMMAND ----------

# MAGIC %md
# MAGIC You may notice that the values in the Series are differently ordered to how they were in the list. This is an inherent feature of Pandas-on-Spark and Pyspark. We will discuss the resaon for this later. 

# COMMAND ----------

type(s)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also create a Pandas-on-Spark DataFrame by passing a dict of objects that can be converted to series-like.

# COMMAND ----------

psdf = ps.DataFrame(
    {'a': [1, 2, 3, 4, 5, 6],
     'b': [100, 200, 300, 400, 500, 600],
     'c': ["one", "two", "three", "four", "five", "six"]},
    index=[10, 20, 30, 40, 50, 60])

psdf

# COMMAND ----------

# MAGIC %md
# MAGIC We create a pandas DataFrame by passing a numpy array, with a datetime index and labeled columns.
# MAGIC We start by creating the datetime index:

# COMMAND ----------

dates = pd.date_range('20130101', periods=6)

dates

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we create the Pandas dataframe filled with random numbers:

# COMMAND ----------

pdf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

pdf

# COMMAND ----------

# MAGIC %md
# MAGIC Now, this pandas DataFrame can be converted to a Pandas-on-Spark DataFrame using `from_pandas()`:

# COMMAND ----------

psdf = ps.from_pandas(pdf)
type(psdf)

# COMMAND ----------

# MAGIC %md
# MAGIC This Pandas-on-Spark dataframe looks and behaves the same as a pandas DataFrame though:

# COMMAND ----------

psdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Also, it is possible to create a Pandas-on-Spark DataFrame from Spark DataFrame.  
# MAGIC 
# MAGIC Creating a Spark DataFrame from pandas DataFrame using `createDataFrame()`:

# COMMAND ----------

sdf = spark.createDataFrame(pdf)

# COMMAND ----------

sdf.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Creating Pandas-on-Spark DataFrame from Spark DataFrame uses the `pandas_api` method.
# MAGIC `pandas_api` is automatically attached to Spark DataFrame and available as an API when Pandas-on-Spark is imported.

# COMMAND ----------

psdf = sdf.pandas_api()

# COMMAND ----------

psdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Pandas-on-Spark Dataframes have specific [dtypes](http://pandas.pydata.org/pandas-docs/stable/basics.html#basics-dtypes).  
# MAGIC Types that are common to both Spark and pandas are currently supported.  

# COMMAND ----------

psdf.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viewing Data
# MAGIC 
# MAGIC See the [docs](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html).

# COMMAND ----------

# MAGIC %md
# MAGIC See the top rows of the `psdf` frame. The results may not be the same as pandas though: unlike pandas, the data in a Spark dataframe is not _ordered_, it has no intrinsic notion of index.
# MAGIC When asked for the head of a dataframe, Spark will just take the requested number of rows from a partition. Do not rely on it to return specific rows, use `.loc` or `iloc` instead.  

# COMMAND ----------

psdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC You can display the index, columns, and the underlying numpy data of the DataFrame.
# MAGIC 
# MAGIC You can also retrieve the index; the index column can be ascribed to a DataFrame, see later.

# COMMAND ----------

psdf.index

# COMMAND ----------

psdf.columns

# COMMAND ----------

nparray = psdf.to_numpy()
print(nparray)

# COMMAND ----------

# MAGIC %md
# MAGIC Describe shows a quick statistic summary of your data:

# COMMAND ----------

psdf.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Transposing your data also works as usual:

# COMMAND ----------

psdf.T

# COMMAND ----------

# MAGIC %md
# MAGIC Of course, you can also sort the Pandas-on-Spark DataFrame, for example by its index:

# COMMAND ----------

psdf.sort_index(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, you can sort by value:

# COMMAND ----------

psdf.sort_values(by='B')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Data
# MAGIC Pandas-on-Spark primarily uses the value `np.nan` to represent missing data. These NaN values are by default not included in computations. 

# COMMAND ----------

pdf1 = pdf.reindex(index=dates[0:4], columns=list(pdf.columns) + ['E'])

# COMMAND ----------

pdf1.loc[dates[0]:dates[1], 'E'] = 1

# COMMAND ----------

psdf1 = ps.from_pandas(pdf1)

# COMMAND ----------

psdf1

# COMMAND ----------

# MAGIC %md
# MAGIC To drop any rows that have missing data.

# COMMAND ----------

psdf1.dropna(how='any')

# COMMAND ----------

# MAGIC %md
# MAGIC Filling missing data.

# COMMAND ----------

psdf1.fillna(value=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stats
# MAGIC Operations in general exclude missing data.
# MAGIC 
# MAGIC Performing a descriptive statistic:

# COMMAND ----------

psdf.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark Configurations
# MAGIC 
# MAGIC Various configurations in PySpark could be applied internally in Pandas-on-Spark.
# MAGIC For example, you can enable Arrow optimization to hugely speed up internal pandas conversion. See <a href="https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html">PySpark Usage Guide for Pandas with Apache Arrow</a>.

# COMMAND ----------

prev = spark.conf.get("spark.sql.execution.arrow.enabled")  # Keep its default value.
ps.set_option("compute.default_index_type", "distributed")  # Use default index prevent overhead.

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings coming from Arrow optimizations.

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", True)
%timeit ps.range(300000).to_pandas()

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", False)
%timeit ps.range(300000).to_pandas()

# COMMAND ----------

ps.reset_option("compute.default_index_type")
spark.conf.set("spark.sql.execution.arrow.enabled", prev)  # Set its default value back.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grouping
# MAGIC By “group by” we are referring to a process involving one or more of the following steps:
# MAGIC 
# MAGIC - Splitting the data into groups based on some criteria
# MAGIC - Applying a function to each group independently
# MAGIC - Combining the results into a data structure

# COMMAND ----------

psdf = ps.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                    'C': np.random.randn(8),
                    'D': np.random.randn(8)})

# COMMAND ----------

psdf

# COMMAND ----------

# MAGIC %md
# MAGIC Grouping and then applying the [sum()](https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.groupby.GroupBy.sum.html?highlight=groupby%20sum#pyspark.pandas.groupby.GroupBy.sum) function to the resulting groups.

# COMMAND ----------

psdf.groupby('A').sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Grouping by multiple columns forms a hierarchical index, and again we can apply the sum function.

# COMMAND ----------

psdf.groupby(['A', 'B']).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotting
# MAGIC See the [docs](https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/frame.html?highlight=plotting#plotting>Plotting).

# COMMAND ----------

pser = pd.Series(np.random.randn(1000),
                 index=pd.date_range('1/1/2000', periods=1000))

# COMMAND ----------

kser = ps.Series(pser)

# COMMAND ----------

kser = kser.cummax()

# COMMAND ----------

kser.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC On a DataFrame, the <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.plot.html?highlight=plot#pyspark.pandas.DataFrame.plot">plot()</a> method is a conventient way to plot all of the columns with labels:

# COMMAND ----------

pdf = pd.DataFrame(np.random.randn(1000, 4), index=pser.index,
                   columns=['A', 'B', 'C', 'D'])

# COMMAND ----------

psdf = ps.from_pandas(pdf)

# COMMAND ----------

psdf = psdf.cummax()

# COMMAND ----------

psdf.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting data in/out
# MAGIC See the <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/io.html?highlight=data%20generator">Input/Output
# MAGIC </a> docs.
# MAGIC 
# MAGIC ##### Let's check first the path to our tmp folder

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

# MAGIC %md
# MAGIC ### CSV
# MAGIC 
# MAGIC CSV is straightforward and easy to use. See <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.to_csv.html?highlight=to_csv#pyspark.pandas.DataFrame.to_csv">here</a> to write a CSV file and <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.read_csv.html?highlight=read_csv#pyspark.pandas.read_csv">here</a> to read a CSV file.

# COMMAND ----------

psdf.to_csv('dbfs:/tmp/foo.csv')
ps.read_csv('dbfs:/tmp/foo.csv').head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parquet
# MAGIC 
# MAGIC Parquet is an efficient and compact file format to read and write faster. See <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.to_parquet.html?highlight=to_parquet#pyspark.pandas.DataFrame.to_parquet">here</a> to write a Parquet file and <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.read_parquet.html?highlight=read_parquet#pyspark.pandas.read_parquet">here</a> to read a Parquet file.

# COMMAND ----------

psdf.to_parquet('dbfs:/tmp/bar.parquet')
ps.read_parquet('dbfs:/tmp/bar.parquet').head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark IO
# MAGIC 
# MAGIC In addition, Pandas-on-Spark fully support Spark's various datasources such as ORC and an external datasource.  See <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.spark.to_spark_io.html?highlight=to_spark_io#pyspark.pandas.DataFrame.spark.to_spark_io">here</a> to write it to the specified datasource and <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.read_spark_io.html?highlight=read_spark_io#pyspark.pandas.read_spark_io">here</a> to read it from the datasource.

# COMMAND ----------

psdf.to_spark_io('dbfs:/tmp/zoo.orc', format="orc")
ps.read_spark_io('dbfs:/tmp/zoo.orc', format="orc").head(10)
