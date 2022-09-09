# Databricks notebook source
# MAGIC %md
# MAGIC # Intro to APEX Platform
# MAGIC 
# MAGIC Demonstrate basic functionality and identify terms related to working in the Databricks workspace.
# MAGIC 
# MAGIC 
# MAGIC ##### Objectives
# MAGIC 1. Execute code in multiple languages
# MAGIC 1. Create documentation cells
# MAGIC 1. Access DBFS (Databricks File System)
# MAGIC 
# MAGIC 
# MAGIC ##### Databricks Notebook Utilities
# MAGIC - <a href="https://docs.databricks.com/notebooks/notebooks-use.html#language-magic" target="_blank">Magic commands</a>: `%python`, `%scala`, `%sql`, `%r`, `%sh`, `%md`
# MAGIC - <a href="https://docs.databricks.com/dev-tools/databricks-utils.html" target="_blank">DBUtils</a>: `dbutils.fs` (`%fs`), `dbutils.notebooks` (`%run`), `dbutils.widgets`
# MAGIC - <a href="https://docs.databricks.com/notebooks/visualizations/index.html" target="_blank">Visualization</a>: `display`, `displayHTML`

# COMMAND ----------

!pip install -U -q awscli

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Execute code in multiple languages
# MAGIC To run the default language of the notebook, no additional commands are needed. 

# COMMAND ----------

print("Run default language")

# COMMAND ----------

# MAGIC %md
# MAGIC To run a different language, it needs to be specified by language magic commands: `%python`, `%scala`, `%sql`, `%r`

# COMMAND ----------

# MAGIC %python
# MAGIC print("Run python")

# COMMAND ----------

# MAGIC %scala
# MAGIC println("Run scala")

# COMMAND ----------

# MAGIC %sql
# MAGIC select "Run SQL"

# COMMAND ----------

# MAGIC %r
# MAGIC print("Run R", quote=FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also run shell commands on the driver using the magic command: `%sh`

# COMMAND ----------

# MAGIC %sh ls -l

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively you can use the exclamation sign `!` to execute shell commands

# COMMAND ----------

!ls -l /
# !ls -l /databricks

# COMMAND ----------

# MAGIC %md
# MAGIC You can render HTML using the function: `displayHTML` (available in Python, Scala, and R)

# COMMAND ----------

html = """<h1 style="color:orange;text-align:center;font-family:Courier">Render HTML</h1>"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create documentation cells
# MAGIC To render a cell as <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">Markdown</a> use the magic command: `%md`  
# MAGIC 
# MAGIC Below are some examples of how you can use Markdown to format documentation. Click this cell and press `Enter` to view the underlying Markdown syntax.
# MAGIC 
# MAGIC 
# MAGIC # Heading 1
# MAGIC ### Heading 3
# MAGIC > block quote
# MAGIC 
# MAGIC 1. **bold**
# MAGIC 2. *italicized*
# MAGIC 3. ~~strikethrough~~
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC - [link](https://www.markdownguide.org/cheat-sheet/)
# MAGIC - `code`
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC   "message": "This is a code block",
# MAGIC   "method": "https://www.markdownguide.org/extended-syntax/#fenced-code-blocks",
# MAGIC   "alternative": "https://www.markdownguide.org/basic-syntax/#code-blocks"
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC ![Spark Logo](https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png)
# MAGIC 
# MAGIC | Element         | Markdown Syntax |
# MAGIC |-----------------|-----------------|
# MAGIC | Heading         | `#H1` `##H2` `###H3` `#### H4` `##### H5` `###### H6` |
# MAGIC | Block quote     | `> blockquote` |
# MAGIC | Bold            | `**bold**` |
# MAGIC | Italic          | `*italicized*` |
# MAGIC | Strikethrough   | `~~strikethrough~~` |
# MAGIC | Horizontal Rule | `---` |
# MAGIC | Code            | ``` `code` ``` |
# MAGIC | Link            | `[text](https://www.example.com)` |
# MAGIC | Image           | `[alt text](image.jpg)`|
# MAGIC | Ordered List    | `1. First items` <br> `2. Second Item` <br> `3. Third Item` |
# MAGIC | Unordered List  | `- First items` <br> `- Second Item` <br> `- Third Item` |
# MAGIC | Code Block      | ```` ``` ```` <br> `code block` <br> ```` ``` ````|
# MAGIC | Table           |<code> &#124; col &#124; col &#124; col &#124; </code> <br> <code> &#124;---&#124;---&#124;---&#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br>|

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Access DBFS (Databricks File System)
# MAGIC The <a href="https://docs.databricks.com/data/databricks-file-system.html" target="_blank">Databricks File System</a> (DBFS) is a virtual file system that allows you to treat cloud object storage as though it were local files and directories on the cluster.
# MAGIC 
# MAGIC To run file system commands on DBFS use the magic command: `%fs`

# COMMAND ----------

# MAGIC %fs mounts

# COMMAND ----------

# MAGIC %fs ls /

# COMMAND ----------

# MAGIC %md
# MAGIC #### List data files in DBFS using magic commands
# MAGIC Use a magic command to display files located in the DBFS directory: **`/databricks/datasets`**

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/

# COMMAND ----------

# MAGIC %fs head /databricks-datasets/README.md

# COMMAND ----------

# MAGIC %md
# MAGIC %fs is shorthand for the DBUtils module: dbutils.fs
# MAGIC 
# MAGIC If you need any help, you can type `help` to recieve further information. 

# COMMAND ----------

# MAGIC %fs help

# COMMAND ----------

# MAGIC %md
# MAGIC #### List data files in DBFS using dbutils
# MAGIC - Use **`dbutils`** to get the files at the directory above and save it to the variable **`files`**
# MAGIC - Use the Databricks ``display()`` function to display the contents in **`files`**

# COMMAND ----------

files = dbutils.fs.ls("/databricks-datasets")   # %fs ls /databricks-datasets
display(files)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inspecting S3 shares using [AWSCLI](https://aws.amazon.com/cli/)

# COMMAND ----------

# !aws s3 ls
!aws s3 ls s3://rbi-apex-at01-ho-dsa-ds-academy-embedded-wave-3/

# COMMAND ----------

!aws s3 ls s3://rbi-apex-at01-ho-dsa-ds-academy-embedded-wave-3/ds_academy_embedded_wave_3
