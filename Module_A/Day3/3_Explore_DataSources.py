# Databricks notebook source
# MAGIC %md
# MAGIC ### Exploring RBI Datasets for [DS Academy Apex workspace](https://wiki.rbinternational.com/confluence/pages/viewpage.action?pageId=462816365)

# COMMAND ----------

# MAGIC %md
# MAGIC Install the AWS tool for finding data

# COMMAND ----------

!pip install awscli

# COMMAND ----------

# MAGIC %md
# MAGIC Use the AWS tool to list the contents of the Use Case bucket

# COMMAND ----------

# MAGIC %md
# MAGIC Attempts to set the region in the notebook:  
# MAGIC + spark.databricks.hive.metastore.glueCatalog.enabled true
# MAGIC + spark.sql.warehouse.dir s3://rbi-apex-at01-ho-dsa-ds-academy-embedded-wave-3/ds_academy_embedded_wave_3
# MAGIC + spark.hadoop.aws.region eu-central-1
# MAGIC + spark.hadoop.fs.s3a.endpoint eu-central-1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring ORBIS Parquet Files

# COMMAND ----------

!aws s3 ls ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ --region eu-central-1

# COMMAND ----------

!aws s3 ls ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ORBIS_API_ORBIS_DESCRIPTION/ --region eu-central-1

# COMMAND ----------

# MAGIC %md
# MAGIC ##### First way to acess:

# COMMAND ----------

spark.sql("SELECT * FROM parquet.`s3://ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ORBIS_API_ORBIS_DESCRIPTION/` LIMIT 10").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Second way to acess:

# COMMAND ----------

spark.read.format("parquet").load("s3://ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ORBIS_API_ORBIS_DESCRIPTION/").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Third way to acess:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM parquet. `s3://ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ORBIS_API_ORBIS_DESCRIPTION/`;

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Creating a local table out of the Parquet files:

# COMMAND ----------

# MAGIC %sql
# MAGIC create table default.orbis 
# MAGIC SELECT * FROM parquet.`s3://ho-adl-pz-orbis-xqwdemgcs5imcuu7ssgme8d3e35xyeuc1a-s3alias/CORPORATE_ORBIS_EXPLORATION/ORBIS_API/ORBIS_API_ORBIS_API/`

# COMMAND ----------

# MAGIC %md
# MAGIC Describing the newly created table

# COMMAND ----------

# MAGIC %sql
# MAGIC describe detail default.orbis

# COMMAND ----------

# MAGIC %md
# MAGIC Querying the table records

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM default.orbis 
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Sesamm Parquet files

# COMMAND ----------

!aws s3 ls s3://ho-adl-pz-sesamm-871po3cydwpi8nfju1j1nz7pcwcdheuc1a-s3alias/SESAMM/ --region eu-central-1

# COMMAND ----------

!aws s3 ls s3://ho-adl-pz-sesamm-871po3cydwpi8nfju1j1nz7pcwcdheuc1a-s3alias/SESAMM/HO_SESAMM/ --region eu-central-1

# COMMAND ----------

!aws s3 ls s3://ho-adl-pz-sesamm-871po3cydwpi8nfju1j1nz7pcwcdheuc1a-s3alias/SESAMM/HO_SESAMM/HO_SESAMM_TIMESERIES/ --region eu-central-1

# COMMAND ----------

# MAGIC %md
# MAGIC Accessing using the first way shown before

# COMMAND ----------

spark.sql("SELECT * FROM parquet.`s3://ho-adl-pz-sesamm-871po3cydwpi8nfju1j1nz7pcwcdheuc1a-s3alias/SESAMM/HO_SESAMM/HO_SESAMM_TIMESERIES/` LIMIT 10").display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Exploring Databricks Open Source Datasets

# COMMAND ----------

!ls /dbfs/databricks-datasets/

# COMMAND ----------

# MAGIC %md
# MAGIC Now you have all the steps to explore any dataset!
