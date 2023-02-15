# Databricks notebook source
# MAGIC %md
# MAGIC # Pandas DataFrame vs. Spark DataFrame 
# MAGIC ## When Parallel Computing Matters
# MAGIC This notebook is based on [this](https://towardsdatascience.com/parallelize-pandas-dataframe-computations-w-spark-dataframe-bba4c924487c) article.

# COMMAND ----------

# MAGIC %md
# MAGIC As an aspiring Data Scientist, you are or will get familiar with popular Python packages such as Numpy, Pandas, Scikit-learn, Keras, and TensorFlow. Together these modules help us extract value from data. 
# MAGIC 
# MAGIC As data continue to become larger and more complex, one other element to consider is a framework dedicated to processing Big Data, such as Apache Spark. In this article, we will demonstrate the capabilities of distributed/cluster computing and present a comparison between the Pandas DataFrame and Spark DataFrame. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas DataFrame
# MAGIC Pandas has become very popular for its ease of use. It utilizes DataFrames to present data in tabular format like a spreadsheet with rows and columns. Importantly, it has very intuitive methods to perform common analytical tasks and a relatively flat learning curve. It loads all of the data into memory on a single machine (one node) for rapid execution. While the Pandas DataFrame has proven to be tremendously powerful in manipulating data, it does have its limits. With data growing at an exponentially rate, complex data processing becomes expensive to handle and causes performance degradation. These operations require parallelization and distributed computing, which the Pandas DataFrame does not support.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introducing Cluster/Distribution Computing and Spark DataFrame
# MAGIC Apache Spark is an open-source cluster computing framework. With cluster computing, data processing is distributed and performed in parallel by multiple nodes. This is recognized as the MapReduce framework because the division of labor can usually be characterized by sets of the map, shuffle, and reduce operations found in functional programming. Spark’s implementation of cluster computing is unique because processes 1) are executed in-memory and 2) build up a query plan which does not execute until necessary (known as lazy execution). Although Spark’s cluster computing framework has a broad range of utility, we only look at the Spark DataFrame for the purpose of this article. Similar to those found in Pandas, the Spark DataFrame has intuitive APIs, making it easy to implement.

# COMMAND ----------

# MAGIC %md
# MAGIC ![Spark Process](https://miro.medium.com/max/1400/1*Op5L-wbDMWrJ2dE8dNAbog.webp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas DataFrame vs. Spark DataFrame
# MAGIC When comparing computation speed between the Pandas DataFrame and the Spark DataFrame, it’s evident that the Pandas DataFrame performs marginally better for relatively small data. With size as the major factor in performance in mind, I conducted a comparison test between the two (script in GitHub). I found that as the size of the data increased, notably beyond 1 millions rows and 1000 columns, the Spark DataFrame can outperform the Pandas DataFrame. Below is an Animated 3D Wireframe Plot to illustrate the comparison for univariate (mean) and bivariate (correlation) calculations. Note that this doesn’t include the high overhead associated with setting up the Spark DataFrame. Additionally, the comparison is conducted on relatively simple operations. In reality, more complex operations are used, which are easier to perform with Pandas DataFrames than with Spark DataFrames.

# COMMAND ----------

# Start Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("demographicsFilter").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC This is very very inefficient and takes a lot of memory. Last time it took over 7 hours with no results (
# MAGIC java.lang.OutOfMemoryError: Java heap space), though it did run once in 41 and once in 45 minutes. I'm not convinced that it is the Pandas part that is the issue, or not only that. Really review the code and best practices of Spark. 
# MAGIC 
# MAGIC Note: The original code had a cutoff once the number of entries (rows * columns) exceeds 1e7. I took that out, but AFTER the failed run, so it is not the erason for the issue. 
# MAGIC 
# MAGIC If we leave that cutoff in, than it might relally be a good idea to plot with a linear regression. Otherwise a huge chunk of the plot is missing. 

# COMMAND ----------

from pyspark.sql.functions import mean, stddev
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
def generate_data(column_counts=[1], row_counts=[1]):
    
    results=[]
    for each_row_count in row_counts: 
        for each_column_count in column_counts: 
            each_element_count = each_row_count*each_column_count
            print(f'{each_column_count} by {each_row_count}')
            result = {}

            result['element_count'] = each_element_count
            result['column_count'] = each_column_count
            result['row_count'] = each_row_count

            # creating array
            ary = np.random.randint(100, size=each_element_count) # generate array of size each_element_count with numbers in [0,100)
            array_time = %timeit -o -n 3 np.random.randint(100, size=each_element_count) # repeat, this time only timing how long creation takes
            result['array_time'] = array_time.best # take smallest time for creating array
            # why this timing? unnecessary? 


            ary = ary.reshape(-1, each_column_count) # reshape <- do this immediately after creating

            # PANDAS
            # create dataframe in pandas
            df = pd.DataFrame(ary) 
            # capture time to covert np array to pandas
            df_create_time = %timeit -o -n 3 pd.DataFrame(ary)
            result['pandas_df_create_time'] = df_create_time.best 
            # capture time to calculate the mean in a pandas dataframe
            df_mean_time = %timeit -o -n 3 df.mean()
            result['pandas_df_mean_time'] = df_mean_time.best 

            # SPARK
            # create dataframe in spark
            spark_df = spark.createDataFrame(df) 
            # capture time to convert np array to spark
            spark_df_create_time = %timeit -o -n 3 spark.createDataFrame(df)
            result['spark_df_create_time'] = spark_df_create_time.best 
            # capture time to calculate the mean in a spark dataframe
            spark_df_mean_time = %timeit -o -n 3 spark_df.select(*[mean(x) for x in spark_df.columns]).collect()
            result['spark_df_mean_time']=spark_df_mean_time.best
            results.append(result)
    return results
# took approx 45 min

# COMMAND ----------


results_large =  generate_data(column_counts=[1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], row_counts=[1, 10, 100, 1000])

# COMMAND ----------


results_small = generate_data(column_counts=[1,10,100], row_counts=[1,10,100])

# COMMAND ----------

import pandas as pd
results = results_small
results_df=pd.DataFrame(results)
#results_df.dropna(inplace=True)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC current status: there are two plots, one showing the time for computing the correlation of a dataframe, one for computing the mean of a dataframe. however, the code for correlation from github does not work. also, there does not seem to be any point where they calculate the mean for a pandas dataframe (only failed attempts for spark). also, at no point are we exporting the data used for the plots, namely results_corr.csv and resutls_mean.csv. However, the mean IS calculated in the main loop.
# MAGIC 
# MAGIC next steps:
# MAGIC - we'll focus on the plotting. we have the information on the time for calculating the mean, so this should be fine for a first attempt. 
# MAGIC 
# MAGIC - once that works, we can try fixing the calculation of the correlation. 
# MAGIC 
# MAGIC - alternatively, we can plot the time it takes to convert a numpy array to the different dataframe types. 

# COMMAND ----------

mean_df = results_df[['array_time', 
                      'column_count', 
                      'row_count', 
                      'element_count', 
                      'pandas_df_create_time', 
                      'pandas_df_mean_time', 
                      'spark_df_create_time', 
                      'spark_df_mean_time']]

# COMMAND ----------

import matplotlib.pyplot as plt

def stacked_line_charts(df, name1, name2, figsize=(12,8)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Comparison of {name1} and {name2}", fontstyle='italic')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey = ax1)
    ax1.set_title('Pandas')
    stacked_plot(df, ax1, name1)
    ax2.set_title('Spark')
    stacked_plot(df, ax2, name2)
    plt.show()

def stacked_plot(df, ax, name):
    for column_count in df['column_count'].unique(): 
        sub_df=df[df['column_count']==column_count]

        ax.plot(np.log10(sub_df['row_count']), 
                 np.log(sub_df[name]), 
                 label=column_count)
    ax.set_ylabel(f"log({name})")
    ax.set_xlabel(f"log(row_count)")
    ax.legend()
    
stacked_line_charts(mean_df, 'pandas_df_create_time', 'spark_df_create_time')

# COMMAND ----------

def scatter_plot(df, title, name1, name2):
    # code is taken pretty 1-1 from github
    # still needs to be prettied up
    
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.scatter3D(np.log(df['row_count']),
                 np.log(df['column_count']), 
                 np.log(df[name1]), 
                 c='tab:red')
    ax.scatter3D(np.log(df['row_count']),
                 np.log(df['column_count']), 
                 np.log(df[name2]), 
                 c='tab:blue')

    all_row_counts=df['row_count'].unique()
    all_column_counts=df['column_count'].unique()
    xlabels_exp=sorted(all_row_counts)
    xlabels_exp_red=[label for idx, label in enumerate(xlabels_exp) if idx%2==0]
    xticks_locs=np.log(xlabels_exp_red)
    xticks_labels=xlabels_exp_red
    ylabels_exp=sorted(all_column_counts)
    yticks_locs=np.log(ylabels_exp)
    yticks_labels=ylabels_exp
    min_time=df[[name1, name2]].min().min()
    max_time=df[[name1, name2]].max().max()
    zlabels_log=np.linspace(np.log(min_time), np.log(max_time), 5)
    zticks_locs=zlabels_log
    zticks_labels=np.exp(zlabels_log).round(2)

    ax.set_xlabel('row_count')
    ax.set_xticks(xticks_locs)
    ax.set_xticklabels(xticks_labels)
    ax.set_ylabel('column_count')
    ax.set_yticks(yticks_locs)
    ax.set_yticklabels(yticks_labels)
    ax.set_zlabel('time (s)')
    ax.set_zticks(zticks_locs)
    ax.set_zticklabels(zticks_labels)
    ax.set_title(title, pad=25)

    plt.close()
    
    animate_3d_plot(ax, fig)

scatter_plot(mean_df, "Comparison", 'pandas_df_create_time', 'spark_df_create_time')

# COMMAND ----------

# MAGIC %md
# MAGIC # Meshgrid

# COMMAND ----------

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# COMMAND ----------

from scipy.interpolate import griddata

def interpolate_data(input_df, name):
    df = input_df[['row_count', 'column_count', name]]
    df.sort_values(['row_count', 'column_count'])
    array = df.to_numpy()
    #array = np.log(array) 
    points = array[:,:2]
    values = array[:,2]
    
    res = 10
    new_x = np.around(np.linspace(np.min(points[:,0]), np.max(points[:,0]), res), 0)
    new_y = np.around(np.linspace(np.min(points[:,1]), np.max(points[:,1]), res), 0)
    xi = cartesian_product(new_x, new_y)
    
    #xi = np.append(points, new_points, axis=0)
    interp_points = griddata(points, array, xi, method='cubic')
    
    new_df = pd.DataFrame(interp_points)
    new_df.dropna(inplace=True) # elements outside convex hull get filled with nan
    new_df.columns = ['row_count', 'column_count', name]
    new_df[['row_count', 'column_count']] = np.around(new_df[['row_count', 'column_count']],0).astype(int)
    new_df[name] = np.around(new_df[name], 7)
    new_df.drop_duplicates(inplace=True)
    
    return new_df

display(interpolate_data(mean_df, "spark_df_create_time"))

# COMMAND ----------

from matplotlib.animation import FuncAnimation

def animate_3d_plot(ax, fig, speed: int = 4):
    plt.close()
    animate = lambda i: ax.view_init(elev=10., azim=i*speed)
    frames = int(360/speed)
    ani = FuncAnimation(fig, animate, frames=frames)
    displayHTML(ani.to_jshtml())
    del ani

# COMMAND ----------


    
def create_mesh_data(input_df, name):
    #get data into correct format
    df = input_df.copy()
    df = df.sort_values(['row_count', 'column_count'])
    data_dict = {(row.row_count, row.column_count): row[name] for index, row in df.iterrows()}

    x_vals = df.row_count.astype(int).unique()
    y_vals = df.column_count.astype(int).unique()
    X,Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            key = (X[i,j],Y[i,j])
            if key in data_dict:
                Z[i,j] = data_dict[key]
            else:
                #Z[i,j] = np.mean(df[name])
                Z[i,j] = 0
    return X,Y,Z

def data_2_logarithmic(X, Y):
    return np.log(X), np.log(Y)

def set_log_scale(ax, x_labels, y_labels):
    ax.set_xticks(np.log(x_labels))
    ax.set_yticks(np.log(y_labels))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
def plot_mesh(input_df, title, name1, name2):
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    
    df1 = interpolate_data(input_df, name1)
    X,Y,Z = create_mesh_data(df1, name1)
    #X, Y = data_2_logarithmic(X, Y)
    ax.plot_wireframe(X, Y, Z, color='c', label=name1)
    
    df2 = interpolate_data(input_df, name2)
    X,Y,Z = create_mesh_data(df2, name2)
    #X, Y = data_2_logarithmic(X, Y)
    ax.plot_wireframe(X, Y, Z, color='r', label=name2)
    
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("row_count")
    ax.set_ylabel("column_count")
    ax.set_zlabel("time")
    #set_log_scale(ax, input_df.row_count, input_df.column_count)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) # make background "transparent"
    animate_3d_plot(ax, fig)
    
#plot_mesh(mean_df, "Compare times to convert numpy array to DataFrame", "pandas_df_create_time", "spark_df_create_time")
plot_mesh(mean_df, "Compare times to calculate mean of Dataframe", "pandas_df_mean_time", "spark_df_mean_time")

# COMMAND ----------

# TO DO
# test everything with more data
# try out cubip interpolation

# COMMAND ----------

# MAGIC %md
# MAGIC ![Correlation and Mean](https://miro.medium.com/max/828/1*fB_N6-Ou9Hy9pL6CCPbQwQ.gif)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC The result is unsurprising given the single node nature of Pandas DataFrames vs. the distributed nature of Spark DataFrames. That is, since execution is done on a single server for the Pandas DataFrame, the in-memory computing speed and capability take a hit for very large data sets. Of course this is contingent upon the capabilities of the hardware.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which is the right DataFrame?
# MAGIC Considering modern hardware specifications and how Pandas optimizes calculations, it is unlikely that a typical exploratory project warrants the implementation of Spark. However, there are cases where Spark makes obvious sense.

# COMMAND ----------

# MAGIC %md
# MAGIC ![Pandas DataFrame vs. Spark DataFrame Characteristics](https://miro.medium.com/max/828/1*udGd83PTRehMda4tSRezcw.webp)

# COMMAND ----------

# MAGIC %md
# MAGIC 1) Spark is useful for applications that require a highly distributed, persistent, and pipelined processing. It might make sense to begin a project using Pandas with a limited sample to explore and migrate to Spark when it matures. This is used today in the development of market trend prediction, personalized customer experience, and weather forecast engines, to name a few.
# MAGIC 2) Spark is useful for Natural Language Processing and Computer Vision applications, which typically entail a great deal of calculations with data that is both wide and long.
# MAGIC 3) Spark has a Machine Learning library, MLlib, which aims to provide high-level APIs to create Machine Learning Pipelines. Machine Learning Pipelines are ideally run on Spark’s framework due to the iterative nature of model tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC As demonstrated in the comparison analysis, the popular Pandas DataFrame is very capable of performing rapid calculation. Although the implementation of Spark DataFrame has become easier since the syntax is intended to match its Pandas counterpart, it’s not always desirable for ad-hoc analysis. It does become very powerful with data processes that are well established, particular when the data is expected to be large.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus
# MAGIC Here’s is the walk-through of how I created the animated 3D Wireframe Plot. At a high level, it can be broken down into the following steps:
# MAGIC 
# MAGIC 1) Capture some data — I used the limited data points to infer additional simulations in steps 2–4
# MAGIC 2) Fit Regression Model to data
# MAGIC 3) Create a grid of the desired X, Y plane
# MAGIC 4) Use Regression Model to calculate Z coordinates for the grid
# MAGIC 5) Generate 3D Wireframe Plot
# MAGIC 6) Repeat step 5 by rotating the axis of the X, Y plane and store each plot as individual PNG files
# MAGIC 7) Use sequential PNG files to generate GIF

# COMMAND ----------


