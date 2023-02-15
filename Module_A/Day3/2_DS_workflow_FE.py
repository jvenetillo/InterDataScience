# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science workflow  
# MAGIC 
# MAGIC In this sequence of notebooks, we will exemplify the inner steps in the Data Science workflow.  
# MAGIC We are not going to discuss the business requirements and deployment strategies, but just the phases below:
# MAGIC 
# MAGIC ###### I - Exploratory Data Analysis 
# MAGIC ##### II - Feature Engineering and Selection (this notebook)  
# MAGIC ###### III - Modeling  
# MAGIC ###### IV - Evaluation  
# MAGIC 
# MAGIC In this notebook, we will use the insights we have gathered previously in the EDA and perform the preparation for the Modeling phase.  
# MAGIC We will perform the following steps:
# MAGIC 1. Data Loading and Cleaning
# MAGIC 2. Feature Selection
# MAGIC 3. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## II - Feature Engineering and Selection  

# COMMAND ----------

# MAGIC %md
# MAGIC The **Feature Engineering and Selection** stage is named "Data Preparation" in CRISP-DM and "Data Engineering" in CRISP-ML:

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **CRISP-DM**
# MAGIC 
# MAGIC 
# MAGIC [CRISP-DM Process](https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png)
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png" width="500" />
# MAGIC 
# MAGIC **CRISP-ML**
# MAGIC 
# MAGIC [CRISP-ML Process](https://ml-ops.org/img/crisp-ml-process.jpg)  
# MAGIC [Source](https://ml-ops.org/content/crisp-ml)
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://ml-ops.org/img/crisp-ml-process.jpg" width="1024" height="512" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Loading and Cleaning
# MAGIC After the data has been analyzed, it is time to clean it. In other words, the data cleaning step will involve handling the problems that were encountered with the data set during analysis. Data professionals will **replace missing values** with neighborhood average / minimum / maximum / any other values in this step. Any other incorrect data points will also be fixed in this step. Data Cleaning also constitutes the removal of outlying data points.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 - Import libraries

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 - Load Data Set and Check Basic Information

# COMMAND ----------

# MAGIC %md
# MAGIC **Visually inspecting the dataset**

# COMMAND ----------

df = pd.read_csv('data/Automobile_data.csv')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **Checking columns and data types**

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC **At this moment, we look for columns that shall be transformed / converted later in the workflow.**

# COMMAND ----------

print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(include='object').columns)
print(df.select_dtypes(include='category').columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Check for missing values
# MAGIC  Here we will repeat the steps from the previous notebook on EDA to remove missing values. Remember that our data set does not have any missing values in the form of NaNs but had wrong entries in the form of symbols such as '?'.

# COMMAND ----------

for col in df.columns:
    df[col].replace({'?': np.nan},inplace=True)
    
# replacing these missing values with mean
num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Outlier detection and removal 
# MAGIC 
# MAGIC ![Outliers](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/Normal%20Distribution%20deviations.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Techniques for outlier detection and removal:
# MAGIC 👉 Z-score treatment :
# MAGIC 
# MAGIC Assumption– The features are normally or approximately normally distributed.
# MAGIC 
# MAGIC 
# MAGIC Step-2: Read and Load the Dataset
# MAGIC 
# MAGIC df = pd.read_csv('placement.csv')
# MAGIC df.sample(5)
# MAGIC Detect and remove outliers data cgpa
# MAGIC 
# MAGIC 
# MAGIC Step-4: Finding the Boundary Values
# MAGIC 
# MAGIC print("Highest allowed",df['cgpa'].mean() + 3*df['cgpa'].std())
# MAGIC print("Lowest allowed",df['cgpa'].mean() - 3*df['cgpa'].std())
# MAGIC Output:
# MAGIC 
# MAGIC Highest allowed 8.808933625397177
# MAGIC Lowest allowed 5.113546374602842
# MAGIC Step-5: Finding the Outliers
# MAGIC 
# MAGIC df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]
# MAGIC Step-6: Trimming of Outliers
# MAGIC 
# MAGIC new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]
# MAGIC new_df
# MAGIC Step-7: Capping on Outliers
# MAGIC 
# MAGIC upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
# MAGIC lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()
# MAGIC Step-8: Now, apply the Capping
# MAGIC 
# MAGIC df['cgpa'] = np.where(
# MAGIC     df['cgpa']>upper_limit,
# MAGIC     upper_limit,
# MAGIC     np.where(
# MAGIC         df['cgpa']<lower_limit,
# MAGIC         lower_limit,
# MAGIC         df['cgpa']
# MAGIC     )
# MAGIC )
# MAGIC Step-9: Now see the statistics using “Describe” Function
# MAGIC 
# MAGIC df['cgpa'].describe()
# MAGIC Output:
# MAGIC 
# MAGIC count    1000.000000
# MAGIC mean        6.961499
# MAGIC std         0.612688
# MAGIC min         5.113546
# MAGIC 25%         6.550000
# MAGIC 50%         6.960000
# MAGIC 75%         7.370000
# MAGIC max         8.808934
# MAGIC Name: cgpa, dtype: float64
# MAGIC This completes our Z-score based technique!
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC 👉 IQR based filtering :
# MAGIC 
# MAGIC Used when our data distribution is skewed.
# MAGIC 
# MAGIC Step-1: Import necessary dependencies
# MAGIC 
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC Step-2: Read and Load the Dataset
# MAGIC 
# MAGIC df = pd.read_csv('placement.csv')
# MAGIC df.head()
# MAGIC Step-3: Plot the distribution plot for the features
# MAGIC 
# MAGIC plt.figure(figsize=(16,5))
# MAGIC plt.subplot(1,2,1)
# MAGIC sns.distplot(df['cgpa'])
# MAGIC plt.subplot(1,2,2)
# MAGIC sns.distplot(df['placement_exam_marks'])
# MAGIC plt.show()
# MAGIC Step-4: Form a Box-plot for the skewed feature
# MAGIC 
# MAGIC sns.boxplot(df['placement_exam_marks'])
# MAGIC Detect and remove outliers boxplot
# MAGIC 
# MAGIC Step-5: Finding the IQR
# MAGIC 
# MAGIC percentile25 = df['placement_exam_marks'].quantile(0.25)
# MAGIC percentile75 = df['placement_exam_marks'].quantile(0.75)
# MAGIC Step-6: Finding upper and lower limit
# MAGIC 
# MAGIC upper_limit = percentile75 + 1.5 * iqr
# MAGIC lower_limit = percentile25 - 1.5 * iqr
# MAGIC Step-7: Finding Outliers
# MAGIC 
# MAGIC df[df['placement_exam_marks'] > upper_limit]
# MAGIC df[df['placement_exam_marks'] < lower_limit]
# MAGIC Step-8: Trimming
# MAGIC 
# MAGIC new_df = df[df['placement_exam_marks'] < upper_limit]
# MAGIC new_df.shape
# MAGIC Step-9: Compare the plots after trimming
# MAGIC 
# MAGIC plt.figure(figsize=(16,8))
# MAGIC plt.subplot(2,2,1)
# MAGIC sns.distplot(df['placement_exam_marks'])
# MAGIC plt.subplot(2,2,2)
# MAGIC sns.boxplot(df['placement_exam_marks'])
# MAGIC plt.subplot(2,2,3)
# MAGIC sns.distplot(new_df['placement_exam_marks'])
# MAGIC plt.subplot(2,2,4)
# MAGIC sns.boxplot(new_df['placement_exam_marks'])
# MAGIC plt.show()
# MAGIC comparison post trimming Detect and remove outliers
# MAGIC 
# MAGIC Step-10: Capping
# MAGIC 
# MAGIC new_df_cap = df.copy()
# MAGIC new_df_cap['placement_exam_marks'] = np.where(
# MAGIC     new_df_cap['placement_exam_marks'] > upper_limit,
# MAGIC     upper_limit,
# MAGIC     np.where(
# MAGIC         new_df_cap['placement_exam_marks'] < lower_limit,
# MAGIC         lower_limit,
# MAGIC         new_df_cap['placement_exam_marks']
# MAGIC     )
# MAGIC )
# MAGIC Step-11: Compare the plots after capping
# MAGIC 
# MAGIC plt.figure(figsize=(16,8))
# MAGIC plt.subplot(2,2,1)
# MAGIC sns.distplot(df['placement_exam_marks'])
# MAGIC plt.subplot(2,2,2)
# MAGIC sns.boxplot(df['placement_exam_marks'])
# MAGIC plt.subplot(2,2,3)
# MAGIC sns.distplot(new_df_cap['placement_exam_marks'])
# MAGIC plt.subplot(2,2,4)
# MAGIC sns.boxplot(new_df_cap['placement_exam_marks'])
# MAGIC plt.show()
# MAGIC comparison post capping
# MAGIC 
# MAGIC This completes our IQR based technique!
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC 👉 Percentile :
# MAGIC 
# MAGIC – This technique works by setting a particular threshold value, which decides based on our problem statement.
# MAGIC 
# MAGIC – While we remove the outliers using capping, then that particular method is known as Winsorization.
# MAGIC 
# MAGIC – Here we always maintain symmetry on both sides means if remove 1% from the right then in the left we also drop by 1%.
# MAGIC 
# MAGIC Step-1: Import necessary dependencies
# MAGIC 
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC Step-2: Read and Load the dataset
# MAGIC 
# MAGIC df = pd.read_csv('weight-height.csv')
# MAGIC df.sample(5)
# MAGIC data height
# MAGIC 
# MAGIC Step-3: Plot the distribution plot of “height” feature
# MAGIC 
# MAGIC sns.distplot(df['Height'])
# MAGIC Step-4: Plot the box-plot of “height” feature
# MAGIC 
# MAGIC sns.boxplot(df['Height'])
# MAGIC boxplot height
# MAGIC 
# MAGIC Step-5: Finding upper and lower limit
# MAGIC 
# MAGIC upper_limit = df['Height'].quantile(0.99)
# MAGIC lower_limit = df['Height'].quantile(0.01)
# MAGIC Step-7: Apply trimming
# MAGIC 
# MAGIC new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]
# MAGIC Step-8: Compare the distribution and box-plot after trimming
# MAGIC 
# MAGIC sns.distplot(new_df['Height'])
# MAGIC sns.boxplot(new_df['Height'])
# MAGIC Detect and remove outliers trriming boxplot
# MAGIC 
# MAGIC 👉 Winsorization :
# MAGIC 
# MAGIC Step-9: Apply Capping(Winsorization)
# MAGIC 
# MAGIC df['Height'] = np.where(df['Height'] >= upper_limit,
# MAGIC         upper_limit,
# MAGIC         np.where(df['Height'] <= lower_limit,
# MAGIC         lower_limit,
# MAGIC         df['Height']))
# MAGIC Step-10: Compare the distribution and box-plot after capping
# MAGIC 
# MAGIC sns.distplot(df['Height'])
# MAGIC sns.boxplot(df['Height'])
# MAGIC boxplot post capping Detect and remove outliers
# MAGIC 
# MAGIC This completes our percentile-based technique!

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.1 Techniques for outlier detection and removal
# MAGIC 
# MAGIC - **Trimming**   
# MAGIC Remove the outliers from the dataset before training a machine learning model. 
# MAGIC 
# MAGIC - **Capping**   
# MAGIC Keep a maximum or minimum threshold and give values to the data points accordingly.
# MAGIC 
# MAGIC - **Discretization**   
# MAGIC This is the method in which numerical features are converted to discrete values using bins. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trimming
# MAGIC 
# MAGIC The trimming method **excludes** the outlier values from our analysis. By applying this technique our **data becomes thin** when there are more outliers present in the dataset. However, trimming is **very fast**, which is the main advantage of this technique.

# COMMAND ----------

def trim_outliers(df, column_name, upper_limit, lower_limit):
    return df[(df[column_name] < upper_limit) & (df[column_name] > lower_limit)]


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Capping
# MAGIC In this technique, any value outside of a particular range will be **set to the minimum or maximum value** allowed. 
# MAGIC Outlier capping reduces the variance of the data **without losing as much information** as when performing trimming. It emphasizes that outliers could be true extremely high or low values, not just errors or pure noise. 

# COMMAND ----------

def cap_outliers(df, column_name, upper_limit, lower_limit):
    new_df = df.copy()
    new_df[column_name] = np.where(
        new_df[column_name] > upper_limit,
        upper_limit,              # values over upper_limit get set to upper_limit
        np.where(                 # values under upper_limit  ...
            new_df[column_name] < lower_limit,
            lower_limit,           # ... get set to lower_limit if they are under lower_limit
            new_df[column_name]   # ... are left untouched if they are between lower_limit and upper_limit
        )
    )
    return new_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Comparison plots
# MAGIC When looking at different methods of outlier treatment, we will want to compare the distribution of data points before and after the respective method has been applied. 

# COMMAND ----------

def plot_feature(df, column_name):
    plt.figure(figsize=(16,5))

    plt.subplot(1,2,1)
    sns.histplot(df[column_name], kde=True)

    plt.subplot(1,2,2)
    sns.boxplot(x=df[column_name])

# COMMAND ----------

def comparison_plot(df1, df2, column_name):

    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    sns.histplot(df1[column_name], kde=True)

    plt.subplot(2,2,2)
    sns.boxplot(x=df1[column_name])

    plt.subplot(2,2,3)
    sns.histplot(df2[column_name], kde=True)

    plt.subplot(2,2,4)
    sns.boxplot(x=df2[column_name])

    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.2 Z-score treatment
# MAGIC 
# MAGIC **Assumption**   
# MAGIC The features are normally or approximately normally distributed.
# MAGIC 
# MAGIC In Z-score treatment, outliers of 3 times the standard deviation are either removed from the dataset, or are capped.   
# MAGIC There is also Z-Score normalization, which additionally transforms the features by subtracting with the mean and normalizing by dividing with the standard deviation. The resulting data will have a mean of 0 and a standard deviation of 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Plot the distribution of the feature

# COMMAND ----------

plot_feature(df, 'stroke')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Finding upper and lower limit

# COMMAND ----------

upper_limit = df['stroke'].mean() + 3*df['stroke'].std()
lower_limit  = df['stroke'].mean() - 3*df['stroke'].std()

print("Highest allowed", upper_limit)
print("Lowest allowed", lower_limit)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.a: Trimming of Outliers

# COMMAND ----------

new_df = trim_outliers(df, 'stroke', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with trimming
comparison_plot(df, new_df, 'stroke')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.b: Capping of Outliers

# COMMAND ----------

new_df = cap_outliers(df, 'stroke', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with capping
comparison_plot(df, new_df, 'stroke')

# COMMAND ----------

# MAGIC %md 
# MAGIC This completes our Z-score based technique!

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.3 IQR based filtering
# MAGIC 
# MAGIC In contrast to Z-score treatment, IQR (interquartile range) based filterting can also be used when our data distribution is skewed. This is because the cutoff for outliers is taken with respect to and proportional to the IQR. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Plot the distribution of the feature

# COMMAND ----------

plot_feature(df, 'engine-size')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Finding the IQR, upper and lower limit

# COMMAND ----------

percentile25 = df['engine-size'].quantile(0.25)
percentile75 = df['engine-size'].quantile(0.75)
iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print("IQR", iqr)
print("Highest allowed", upper_limit)
print("Lowest allowed", lower_limit)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.a: Trimming outliers

# COMMAND ----------

new_df = trim_outliers(df, 'engine-size', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with trimming
comparison_plot(df, new_df, 'engine-size')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.b: Capping Outliers

# COMMAND ----------

new_df = cap_outliers(df, 'engine-size', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with capping
comparison_plot(df, new_df, 'engine-size')

# COMMAND ----------

# MAGIC %md
# MAGIC This completes our IQR based technique!

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4.4 Percentile based techniques
# MAGIC The percentile technique works by removing a particular percentage of points from the lower and upper end of the data distribution. THe precise threshold is decided based on the problem statement. 
# MAGIC 
# MAGIC + As the outliers are removed, this technique applies *trimming*. When using capping, this method is called **Winsorization**.
# MAGIC 
# MAGIC + This technique always maintains symmetry on both sides, since removal of 1% from the right side also means we drop 1% from the left. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1: Plot the distribution of the feature before and after outlier removal

# COMMAND ----------

plot_feature(df, 'stroke')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2: Finding upper and lower limit

# COMMAND ----------

percentile = 0.01
upper_limit = df['length'].quantile(1-percentile)
lower_limit = df['length'].quantile(percentile)

print("Highest allowed", upper_limit)
print("Lowest allowed", lower_limit)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.a: Trimming of Outliers

# COMMAND ----------

new_df = trim_outliers(df, 'length', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with trimming
comparison_plot(df, new_df, 'length')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 3.b: Capping of Outliers (Winsorization)

# COMMAND ----------

new_df = cap_outliers(df, 'length', upper_limit, lower_limit)

# Plot the distribution of the feature before and after outlier removal with capping
comparison_plot(df, new_df, 'length')

# COMMAND ----------

# MAGIC %md
# MAGIC This completes our percentile-based technique!

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Selection
# MAGIC Now that the dataset is clean, it is time to prepare a feature subspace containing features relevant to the model. The first step toward preparing this subspace is to use basic logic and reasoning to pick the relevant features. Then, you can use popular feature selection methods in ML: intrinsic methods, wrapper methods, or filter methods, depending on your dataset characteristics.
# MAGIC 
# MAGIC ## 3. Feature Engineering
# MAGIC You will often find variables in your dataset that can be converted into useful features using encoding methods. For example, the color of a flower can be used to distinguish it from others. But, as color is usually a string type of variable, it cannot be directly fed to a machine learning model. Using feature engineering techniques in such cases can prove to be useful. If you explore enough projects in data science, you will find feature engineering methods are commonly used for categorical feature variables.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## III - Modeling  

# COMMAND ----------

# MAGIC %md This will be developed in the next modules

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## IV - Evaluation  

# COMMAND ----------

# MAGIC %md This will be developed in the next modules
