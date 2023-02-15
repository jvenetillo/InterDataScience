# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Intro to BambooLib
# MAGIC 
# MAGIC [Source](https://bamboolib.8080labs.com/)

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook gives a very short glimpse at a package called **bamboolib**.  
# MAGIC 
# MAGIC Bamboolib is a library for Python that allows you to create and edit dataframes using a graphical user interface (GUI) in your notebook. It allows you to perform common data cleaning and manipulation tasks, such as sorting, filtering, and adding or deleting columns, without writing any code.
# MAGIC 
# MAGIC Bamboolib is built on top of the Pandas library, and it allows you to use the full power of Pandas while still benefiting from the convenience of a GUI. It also includes a number of additional features, such as automatic backup of dataframes, support for undo/redo, and integration with popular visualization libraries like Matplotlib and Seaborn.
# MAGIC 
# MAGIC Bamboolib is integrated with Databricks, allowing for a user-friendly exploration of the DBFS, as well as other useful features. 

# COMMAND ----------

# Run this cell if you need to install Bamboolib.
# You can also install bamboolib on the cluster. Just talk to your cluster admin for that.

%pip install bamboolib

# COMMAND ----------

import pandas as pd
import bamboolib as bam

# COMMAND ----------

# MAGIC %md
# MAGIC The comment `bam` opens a UI from which you can import your data.

# COMMAND ----------

# This opens the UI 
bam

# COMMAND ----------

# MAGIC %md
# MAGIC To see the functionality of bamboolib, we use the winequality dataset to explore the Bamboolib functionality.

# COMMAND ----------

import pandas as pd; import numpy as np
df = pd.read_csv(r'/dbfs/databricks-datasets/wine-quality/winequality-red.csv', sep=';', decimal='.', nrows=100000)
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Have a go at playing around with the different functionalities the UI offers:

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC For more information, take a look at the [bamboolib documentation](https://docs.bamboolib.8080labs.com/).
