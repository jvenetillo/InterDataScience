# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Intro to BambooLib
# MAGIC 
# MAGIC [Source](https://bamboolib.8080labs.com/)

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook gives a very short glimpse at a package called **bamboolib**.  
# MAGIC bamboolib is a Graphical User Interface (GUI) for Pandas DataFrames.
# MAGIC 
# MAGIC It provides an intuitive GUI that exports Python code and supports all common transformations and visualizations.
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

df
