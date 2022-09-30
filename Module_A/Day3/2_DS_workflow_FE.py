# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science workflow  
# MAGIC 
# MAGIC In this sequence of notebooks, we will exemplify the inner steps in the Data Science workflow.  
# MAGIC We are not going to discuss the business requirements and deployment strategies, but just the phases below:
# MAGIC 
# MAGIC ##### I - Exploratory Data Analysis 
# MAGIC ### II - Feature Engineering and Selection (this notebook)  
# MAGIC ##### III - Modeling  
# MAGIC ##### IV - Evaluation  
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
# MAGIC ##### Feature Engineering and Selection is named "Data Preparation" in CRISP-DM and "Data Engineering" in CRISP-ML:

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## CRISP-DM
# MAGIC 
# MAGIC 
# MAGIC [CRISP-DM Process](https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png)
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png" width="768" height="512" />
# MAGIC 
# MAGIC ## CRISP-ML
# MAGIC 
# MAGIC [CRISP-ML Process](https://ml-ops.org/img/crisp-ml-process.jpg)  
# MAGIC [Source](https://ml-ops.org/content/crisp-ml)
# MAGIC 
# MAGIC <br>
# MAGIC <img src="https://ml-ops.org/img/crisp-ml-process.jpg" width="1024" height="512" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Loading and Cleaning
# MAGIC After the data has been analyzed, it is time to clean it. In other words, the data cleaning step will involve handling the problems that were encountered with the dataset during analysis. So, data professionals will replace missing values with neighborhood average/minimum/maximum/ any other values in this step. Any other incorrect data points will also be fixed in this step. Data Cleaning also constitutes the removal of outlying data points.
# MAGIC 
# MAGIC ### 1.1 - Import libraries

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 - Load Dataset and check basic info

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Visually inspecting the dataset

# COMMAND ----------

df = pd.read_csv('data/Automobile_data.csv')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking columns and data types

# COMMAND ----------

#df.columns
df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### At this moment, you look for columns that shall be transformed/converted later in the workflow.

# COMMAND ----------

print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(include='object').columns)
print(df.select_dtypes(include='category').columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Check for missing values

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC It seems there are not missing values, but that may be misleading. Let's explore a bit more:

# COMMAND ----------

#Checking for wrong entries like symbols -,?,#,*,etc.
for col in df.columns:
    print('{} : {}'.format(col, df[col].unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC There are null values in our dataset in form of ‘?’ only but Pandas is not recognizing them so we will replace them into np.nan form.

# COMMAND ----------

for col in df.columns:
    df[col].replace({'?': np.nan},inplace=True)
    
df.info()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 3.1 Visualizing the missing values  
# MAGIC Now the missing values are identified in the dataframe. With the help of heatmap, we can see the amount of data that is missing from the attribute. With this, we can make decisions whether to drop these missing values or to replace them. Usually dropping the missing values is not advisable but sometimes it may be helpful too.

# COMMAND ----------

plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC Now observe that there are many missing values in normalized_losses while other columns have fewer missing values. We can’t drop the normalized_losses column as it may be important for our prediction.  
# MAGIC We can also use the **missingno** libray for a better evaluation of the missing values. First we can check the quantity and how they distribute among the rows:

# COMMAND ----------

import missingno as msno

# COMMAND ----------

msno.bar(df)

# COMMAND ----------

msno.matrix(df)

# COMMAND ----------

# MAGIC %md
# MAGIC The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another  

# COMMAND ----------

msno.heatmap(df)

# COMMAND ----------

# MAGIC %md
# MAGIC The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap  

# COMMAND ----------

msno.dendrogram(df)

# COMMAND ----------

# MAGIC %md  
# MAGIC 
# MAGIC #### 3.2. Replacing the missing values
# MAGIC We will be replacing these missing values with mean because the number of missing values is not great (we could have used the median too).  
# MAGIC Later, in the data preparation phase, we will learn other imputation techniques.  

# COMMAND ----------

df.select_dtypes(include='number').head()

# COMMAND ----------

df.select_dtypes(include='object').head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's transform the mistaken datatypes for numeric values and fill with the mean, using the strategy we have chosen.  

# COMMAND ----------

num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Checking for outliers
# MAGIC 
# MAGIC ![Outliers](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/Normal%20Distribution%20deviations.png)

# COMMAND ----------

Techniques for outlier detection and removal:
👉 Z-score treatment :

Assumption– The features are normally or approximately normally distributed.


Step-2: Read and Load the Dataset

df = pd.read_csv('placement.csv')
df.sample(5)
Detect and remove outliers data cgpa


Step-4: Finding the Boundary Values

print("Highest allowed",df['cgpa'].mean() + 3*df['cgpa'].std())
print("Lowest allowed",df['cgpa'].mean() - 3*df['cgpa'].std())
Output:

Highest allowed 8.808933625397177
Lowest allowed 5.113546374602842
Step-5: Finding the Outliers

df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]
Step-6: Trimming of Outliers

new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]
new_df
Step-7: Capping on Outliers

upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()
Step-8: Now, apply the Capping

df['cgpa'] = np.where(
    df['cgpa']>upper_limit,
    upper_limit,
    np.where(
        df['cgpa']<lower_limit,
        lower_limit,
        df['cgpa']
    )
)
Step-9: Now see the statistics using “Describe” Function

df['cgpa'].describe()
Output:

count    1000.000000
mean        6.961499
std         0.612688
min         5.113546
25%         6.550000
50%         6.960000
75%         7.370000
max         8.808934
Name: cgpa, dtype: float64
This completes our Z-score based technique!

 

👉 IQR based filtering :

Used when our data distribution is skewed.

Step-1: Import necessary dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Step-2: Read and Load the Dataset

df = pd.read_csv('placement.csv')
df.head()
Step-3: Plot the distribution plot for the features

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])
plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])
plt.show()
Step-4: Form a Box-plot for the skewed feature

sns.boxplot(df['placement_exam_marks'])
Detect and remove outliers boxplot

Step-5: Finding the IQR

percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)
Step-6: Finding upper and lower limit

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
Step-7: Finding Outliers

df[df['placement_exam_marks'] > upper_limit]
df[df['placement_exam_marks'] < lower_limit]
Step-8: Trimming

new_df = df[df['placement_exam_marks'] < upper_limit]
new_df.shape
Step-9: Compare the plots after trimming

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])
plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])
plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])
plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])
plt.show()
comparison post trimming Detect and remove outliers

Step-10: Capping

new_df_cap = df.copy()
new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)
Step-11: Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])
plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])
plt.show()
comparison post capping

This completes our IQR based technique!

 

👉 Percentile :

– This technique works by setting a particular threshold value, which decides based on our problem statement.

– While we remove the outliers using capping, then that particular method is known as Winsorization.

– Here we always maintain symmetry on both sides means if remove 1% from the right then in the left we also drop by 1%.

Step-1: Import necessary dependencies

import numpy as np
import pandas as pd
Step-2: Read and Load the dataset

df = pd.read_csv('weight-height.csv')
df.sample(5)
data height

Step-3: Plot the distribution plot of “height” feature

sns.distplot(df['Height'])
Step-4: Plot the box-plot of “height” feature

sns.boxplot(df['Height'])
boxplot height

Step-5: Finding upper and lower limit

upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)
Step-7: Apply trimming

new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]
Step-8: Compare the distribution and box-plot after trimming

sns.distplot(new_df['Height'])
sns.boxplot(new_df['Height'])
Detect and remove outliers trriming boxplot

👉 Winsorization :

Step-9: Apply Capping(Winsorization)

df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))
Step-10: Compare the distribution and box-plot after capping

sns.distplot(df['Height'])
sns.boxplot(df['Height'])
boxplot post capping Detect and remove outliers

This completes our percentile-based technique!

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
