# Databricks notebook source
# MAGIC %md
# MAGIC # Regression Models
# MAGIC 
# MAGIC ## Numerical data with regularization

# COMMAND ----------

# MAGIC %md
# MAGIC **nspired by [this repo](https://github.com/Weesper1985/Predict_real_estate_prices), codes [here](https://www.kaggle.com/dhainjeamita/linear-regression/code), [here](https://www.kaggle.com/vikrishnan/house-sales-price-using-regression) and [here](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/). Used [this data](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime). Tried using [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html) techniques**

# COMMAND ----------

import os
import sys
import time
import pickle
import itertools
import pandas as pd
import numpy as np
import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 25, 25

from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
import seaborn as sns
#import pydotplus

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from IPython.display import Image  


# COMMAND ----------

#Checking if we have a GPU
device_lib.list_local_devices()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 - Importing the datasets and exploring data

# COMMAND ----------

datapath = "./data/communities/"
outputs = "./data/communities/"

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's take a look at what files we have to work with.

# COMMAND ----------

!ls ./data/communities/

# COMMAND ----------

! head -n10 ./data/communities/communities.names

# COMMAND ----------

! head -n3 ./data/communities/communities.data

# COMMAND ----------

# MAGIC %md
# MAGIC It appears the file communities.names contains information about our dataset, whereas communities.data contains tabular data. However, the second does not appear to have any table headers. If we explore the communities.names file further, we can find this missing information:

# COMMAND ----------

! tail -n+72 ./data/communities/communities.names | head -n-320

# COMMAND ----------

# MAGIC %md 
# MAGIC The column headers are indicated by '@attribute'. We will extract this information and merge it with the rest of the data. 

# COMMAND ----------

names = open(os.path.join(datapath, 'communities.names'), 'r').readlines()
attributes = [n for n in names if n.startswith('@attribute')]
attributes = [a.split()[1] for a in attributes]

df = pd.read_csv(os.path.join(datapath, 'communities.data'), 
                 na_values='?', 
                 header=None, 
                 names=attributes)
df.head()

# COMMAND ----------

df.info(verbose=True, null_counts=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 - EDA
# MAGIC As the purpose of this notebook is to demonstrate different regression models, we will keep the EDA short. 

# COMMAND ----------

# non numeric
columns1 = ['state', 'county', 'community', 'communityname', 'fold']

# many missing values
columns2 = ['LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 
            'LemasTotalReq',  'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 
            'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 
            'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
            'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
            'LemasGangUnitDeploy', 'PolicBudgPerPop', 'OtherPerCap']

df.drop(columns=columns1, inplace=True)
df.drop(columns=columns2, inplace=True)

# histograms
df.hist(bins=10,figsize=(30,30),grid=False);

# COMMAND ----------



# COMMAND ----------

df.plot(kind='box', subplots=True, layout=(10,10), sharex=False, sharey=False, figsize=(16,12))
plt.show()

# COMMAND ----------

# correlation
pd.set_option('precision', 2)
df.corr(method='pearson')

# COMMAND ----------

corr=df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True, annot=False, cmap='YlGnBu', linecolor="white")
plt.title('Correlation between features');

# COMMAND ----------

crimes = df['ViolentCrimesPerPop']

# COMMAND ----------

len(df.columns)

# COMMAND ----------

def scatter_with_regression_line(df, first_index):
    plt.figure(figsize=(20, 5))

    for i, col in enumerate(df.columns[first_index:first_index+5]):
        plt.subplot(1, 5, i+1)
        x = df[col]
        y = crimes
        plt.plot(x, y, 'o')
        
        # Create regression line
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('crimes')

# COMMAND ----------

for i in range(20):
    scatter_with_regression_line(df, i*5)


# COMMAND ----------

# MAGIC %md
# MAGIC The point here is to test 'crimes' in a very lean way. 
# MAGIC We'll do this paying attention to: 
# MAGIC - Histogram - Kurtosis and skewness. 
# MAGIC - Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

# COMMAND ----------

from scipy import stats

#histogram and normal probability plot
sns.histplot(crimes, kde = True);
fig = plt.figure()
res = stats.probplot(crimes, plot=plt)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 - Modelling

# COMMAND ----------

# Test options and evaluation metric using Root Mean Square error method
#http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

num_folds = 10
score = 'r2'

# COMMAND ----------

def grid_search(model, param_grid, num_folds, score, X, y):
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, cv=kfold)
    grid_result = grid.fit(X, y)
    return grid_result

# COMMAND ----------

def kfold_cross_validation(pipelines, num_folds, score, X, y):
    results = []
    names = []
    str_len = max([len(name) for name, model in pipelines]) + 1
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=score)
        results.append(cv_results)
        names.append(name)
        msg = f"{name+':':{str_len}s} {cv_results.mean(): 1.3f} (+- {cv_results.std():1.3f})"
        print(msg)
    return names, results

# COMMAND ----------

def model_boxplot_comparison(title, results, names, figsize=(12,8)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.show()
   

# COMMAND ----------

def print_grid_result(grid_result):
    print(f"Best: {grid_result.best_score_:1.3f} using {grid_result.best_params_}\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean:1.3f} ({stdev:1.3f}) with: {param}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 - Data Preperation

# COMMAND ----------

# split into input (X) and output (Y) variables
X = df.values[:,0:-1]
y = df.values[:,99]

# Split-out validation dataset
validation_size = 0.20
seed = 0
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# Rescale features for better performance
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_validation)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - Spot Check Algorithms
# MAGIC A spot-checking algorithms is designed to quickly provide a first set of results on a new predictive model.
# MAGIC 
# MAGIC Unlike grid searching and other types of algorithm tuning that seek the optimal configuration for an algorithm, spot-checking is intended to evaluate a large set of algorithms rapidly for a first result. 

# COMMAND ----------

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso(random_state=seed)))
models.append(('EN', ElasticNet(random_state=seed)))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor(random_state=seed)))
models.append(('SVR', SVR()))
models.append(('RIDGE', Ridge(random_state=seed)))
models.append(('RIDGECV', RidgeCV()))
models.append(('MLPR', MLPRegressor(random_state=seed)))
models.append(('GPR', GaussianProcessRegressor(random_state=seed)))

# COMMAND ----------

names, results = kfold_cross_validation(models, num_folds, score, X_train, Y_train)

# COMMAND ----------

model_boxplot_comparison('Algorithm Comparison', results, names)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 - Spot Check with Scaled Dataset

# COMMAND ----------

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso(random_state=seed))])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet(random_state=seed))])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor(random_state=seed))])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', StandardScaler()),('RIDGE', Ridge(random_state=seed))])))
pipelines.append(('ScaledRIDGECV', Pipeline([('Scaler', StandardScaler()),('RIDGECV', RidgeCV())])))
pipelines.append(('ScaledMLPR', Pipeline([('Scaler', StandardScaler()),('MLPR', MLPRegressor(random_state=seed))])))
pipelines.append(('ScaledGPR', Pipeline([('Scaler', StandardScaler()),('GPR', GaussianProcessRegressor(random_state=seed))])))


names, results = kfold_cross_validation(pipelines, num_folds, score, X_train, Y_train)

# COMMAND ----------

model_boxplot_comparison('Scaled Algorithm Comparison', results, names)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 - Tuning Models

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 - KNN Algorithm tuning

# COMMAND ----------

k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 - Regression Tree Algorithm tuning

# COMMAND ----------

min_samples_split_values = [2,3,4]
min_samples_leaf_values = [1,20,40,60,80]

param_grid = dict(min_samples_split=min_samples_split_values, min_samples_leaf=min_samples_leaf_values)
model = DecisionTreeRegressor(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2.1 - Interpreting Regression Trees [(see here)](http://blog.datadive.net/interpreting-random-forests/)
# MAGIC 
# MAGIC A Decision Tree is a tree in which the nodes represent decisions, and the edges or branches are binary (yes/no, true/false) representing possible paths from one node to another. To use a decision tree for classification or regression, we grab one row of data or a set of features and starts at the root. Through each subsequent decision node we arrive at the terminal node. The process is very intuitive and easy to interpret, which allows trained decision trees to be used for variable selection or more generally, feature engineering.

# COMMAND ----------

!pip install treeinterpreter

# COMMAND ----------

# https://github.com/andosa/treeinterpreter
from treeinterpreter import treeinterpreter as ti

# COMMAND ----------

def tree_interpretation(bias, contributions, figsize=(10,8)):
    #prediction, bias, contributions = ti.predict(model, sample.T)
    
    ft_list = []
    print("Sample")
    print(f"Bias (trainset mean):    {bias[0]:.3f}\n")
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[0], df.columns[:-1]), key=lambda x: -abs(x[0])):
        if abs(round(c, 2)) < 1e-2:
                break
        ft_list.append((feature, round(c, 2)))
        print(f"{feature+':':22s}  {c: .3f}")
        
    print("All other features do not significantly contribute.")
    
    print(f"\n\nPrediction = {prediction[0][0]:.3f}\n")
    print(f"Sum of feature contributions = {np.sum(contributions, axis=1)[0]:.3f}")
    print(f"Bias + contributions = {bias[0]+np.sum(contributions, axis=1)[0]:.3f}")
    
    
    
    # plot
    labels, values = zip(*ft_list)
    xs = np.arange(len(labels)) 

    fig, ax = plt.subplots(figsize=figsize)
    plt.plot((-1,len(ft_list)),(0,0),c="black")
    plt.bar(xs, values, align='center')
    plt.xticks(xs, labels, rotation=-45)

    plt.yticks(values)
    plt.show()
    

# COMMAND ----------

# MAGIC %md
# MAGIC The TreeInterpreter library **decomposes the predictions** as the **sum of contributions from each feature**.  
# MAGIC I.e. prediction = bias + feature 1 contribution + … + feature n contribution.  

# COMMAND ----------

dt = DecisionTreeRegressor(random_state=seed,
                           min_samples_leaf=40,
                           min_samples_split=2).fit(rescaledX, Y_train)

sample = X_validation[100].reshape(-1, 1) # single observation
prediction, bias, contributions = ti.predict(dt, sample.T)

tree_interpretation(bias, contributions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 - Lasso Algorithm Tuning

# COMMAND ----------

alpha_values = np.linspace(0.0004,0.002,15)
param_grid = dict(alpha=alpha_values)
model = Lasso(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 - Ridge Algorithm Tuning

# COMMAND ----------

alpha_values = [0, 1, 10, 20, 50, 80, 100, 120, 200, 500]
param_grid = dict(alpha=alpha_values)
model = Ridge(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 - ElasticNet Algorithm Tuning

# COMMAND ----------

l1_values = np.linspace(1e-07,1e-05, 15)
param_grid = dict(l1_ratio=l1_values)
model = ElasticNet(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 - MLP Algorithm Tunning

# COMMAND ----------

solver_values = ['lbfgs'] #, 'sgd', 'adam']
alpha_values = np.linspace(1e-5, 1e-4, 10)

param_grid = dict(solver=solver_values, alpha=alpha_values)
model = MLPRegressor(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, X_train, Y_train) # Without scaling, that seems not to improve model

# can take >= 5 min

# COMMAND ----------

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 - GPR Algorithm Tunning

# COMMAND ----------

alpha_values = np.linspace(1e-8, 1e-9, 6)
kernel_values = [1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
                 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=2, noise_level_bounds=(1e-10, 1e+1))]
param_grid = dict(kernel=kernel_values, alpha=alpha_values)
model = GaussianProcessRegressor(random_state=seed)

grid_result = grid_search(model, param_grid, num_folds, score, X_train, Y_train) # Without scaling, that does not seem to improve the model

# can take >= 20 min

# COMMAND ----------

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 - Plotting the tuned algorithms

# COMMAND ----------

pipelines = []
pipelines.append(('Scaled LR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('Scaled Opt LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso(alpha=0.0012, random_state=seed))])))
pipelines.append(('Scaled Opt EN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet(l1_ratio=1e-07, random_state=seed))])))
pipelines.append(('Scaled Opt KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor(n_neighbors=11))])))
pipelines.append(('Scaled Opt CART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor(min_samples_leaf=40,
                                                                                                           min_samples_split=3,
                                                                                                           random_state=seed))])))
pipelines.append(('Scaled SVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
pipelines.append(('Scaled Opt RIDGE', Pipeline([('Scaler', StandardScaler()),('RIDGE', Ridge(alpha=80, random_state=seed))])))
pipelines.append(('Scaled RIDGECV', Pipeline([('Scaler', StandardScaler()),('RIDGECV', RidgeCV())])))
pipelines.append(('Opt MLPR', Pipeline([('MLPR', MLPRegressor(alpha=5e-05, solver='lbfgs', random_state=seed))])))
pipelines.append(('Opt GPR', Pipeline([('GPR', GaussianProcessRegressor(alpha=1e-8, 
                                                                        kernel=1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
                                                                        + WhiteKernel(noise_level=2, noise_level_bounds=(1e-10, 1e+1)), 
                                                                        random_state=seed))])))

names, results = kfold_cross_validation(pipelines, num_folds, score, X_train, Y_train)

# COMMAND ----------

model_boxplot_comparison('Scaled and Optimized Algorithm Comparison', results, names)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.9 - Ensemble Models

# COMMAND ----------

ensembles = []
ensembles.append(('Scaled AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor(random_state=seed))])))
ensembles.append(('Scaled GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor(random_state=seed))])))
ensembles.append(('Scaled RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(random_state=seed))])))
ensembles.append(('Scaled ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(random_state=seed))])))

names_ensembles, results_ensembles = kfold_cross_validation(ensembles, num_folds, score, X_train, Y_train)

# COMMAND ----------

model_boxplot_comparison('Scaled Ensemble Algorithm Comparison', results_ensembles, names_ensembles)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.9.1 - AdaBoost Algorithm Tuning

# COMMAND ----------

param_grid = dict(n_estimators=np.array([2,5,10,20,30,40]))
model = AdaBoostRegressor(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)
    
print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.9.2 - GBM Algorithm Tuning

# COMMAND ----------

param_grid = dict(n_estimators=np.array([20,35,40,45,50,55,60,100]))
model = GradientBoostingRegressor(random_state=seed)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.9.3 - Random Forest Algorithm Tuning

# COMMAND ----------

n_estimators_values=np.array([30,50,70])
min_samples_split_values = [2,3]
min_samples_leaf_values = [10,20,30]

param_grid = dict(n_estimators=n_estimators_values,
                  min_samples_split=min_samples_split_values, 
                  min_samples_leaf=min_samples_leaf_values)

model = RandomForestRegressor(random_state=seed, n_jobs=-1)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)

# COMMAND ----------

# MAGIC %md
# MAGIC For the Random Forest, similarly to the Descision Tree Regressor, we will take a look at how the different features contribute to the prediction on a sample. 

# COMMAND ----------

rf = RandomForestRegressor(random_state=seed, 
                           n_jobs=-1,
                           n_estimators=50,
                           min_samples_leaf=1,
                           min_samples_split=2).fit(rescaledX, Y_train)

sample = X_validation[100].reshape(-1, 1) # single observation
prediction, bias, contributions = ti.predict(rf, sample.T)
tree_interpretation(bias, contributions, figsize=(15,8))


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.9.4 - Extra Tree Algorithm Tuning

# COMMAND ----------

n_estimators_values=np.array([200,300])
min_samples_split_values = [2,3]
min_samples_leaf_values = [1,10,20]

param_grid = dict(n_estimators=n_estimators_values,
                  min_samples_split=min_samples_split_values, 
                  min_samples_leaf=min_samples_leaf_values)

model = ExtraTreesRegressor(random_state=seed, n_jobs=-1)
grid_result = grid_search(model, param_grid, num_folds, score, rescaledX, Y_train)

print_grid_result(grid_result)    

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.9.5 - Optimized Ensembles

# COMMAND ----------

ensembles = []
ensembles.append(('Scaled Opt AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor(n_estimators=5))])))
ensembles.append(('Scaled Opt GBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor(n_estimators=60))])))
ensembles.append(('Scaled Opt RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=70, 
                                                                                                        min_samples_split=3,
                                                                                                        min_samples_leaf=10))])))
ensembles.append(('Scaled Opt ET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=300, 
                                                                                                      min_samples_split=3,
                                                                                                      min_samples_leaf=1))])))

names_ensembles, results_ensembles = kfold_cross_validation(ensembles, num_folds, score, X_train, Y_train)
    

# COMMAND ----------

model_boxplot_comparison('Scaled Optimized Ensemble Algorithm Comparison', results_ensembles, names_ensembles, figsize=(8,6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.10 - Summarizing all the optimized results

# COMMAND ----------

results_all = results + results_ensembles
names_all = names + names_ensembles

print(pd.Series([x.mean() for x in results_all], index=names_all).sort_values(ascending=False))

model_boxplot_comparison('All Algorithms Comparison', results_all, names_all, figsize=(20,12))

# COMMAND ----------

# MAGIC %md
# MAGIC **So our best regressors are Gaussian Processes, Extra Trees, Ridge and Lasso**
# MAGIC 
# MAGIC Let's examine how Lasso / Ridge / ElasticNet change with regularization
# MAGIC i.e. examine how eliminating irrelevant features affects the models, by setting their coeficients to zero or close:

# COMMAND ----------

# prepare the model
model = Lasso(alpha=0.00145, random_state=seed)
model.fit(rescaledX, Y_train)

# COMMAND ----------

eval = pd.Series(model.coef_, index=df.columns[:-1]).sort_values()
eval

# COMMAND ----------

eval.plot(kind='bar', title='Modal Coefficients', figsize=(12,8))

# COMMAND ----------

eval2 = eval[np.abs(eval)>1e-6]
eval2.plot(kind='bar', title='Modal Coefficients', figsize=(12,8))

# COMMAND ----------

# MAGIC %md
# MAGIC **Verifying the Lasso and ElasticNet paths**
# MAGIC 
# MAGIC (see pages pages 69-73 from [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

# COMMAND ----------

def compare_paths(paths, coefs, names, figsize=(12,8)):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_0 = -np.log10(paths[0])
    neg_log_1 = -np.log10(paths[1])
    for coef_0, coef_1, c in zip(coefs[0], coefs[1], colors):
        l0 = plt.plot(neg_log_0, coef_0, c=c)
        l1 = plt.plot(neg_log_1, coef_1, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title(f'{names[0]} and {names[1]} Paths')
    plt.legend((l0[-1], l1[-1]), names, loc='lower left')
    plt.axis('tight')

# COMMAND ----------

eps = 5e-3

# Computing regularization path using the lasso
alphas_lasso, coefs_lasso, _ = lasso_path(rescaledX, Y_train, eps=eps, random_state=0)

# Computing regularization path using the positive lasso
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(rescaledX, Y_train, eps=eps, positive=True, fit_intercept=False)

# Computing regularization path using the elastic net
alphas_enet, coefs_enet, _ = enet_path(rescaledX, Y_train, eps=eps, l1_ratio=0.8, fit_intercept=False)

# Computing regularization path using the positive elastic net
alphas_positive_enet, coefs_positive_enet, _ = enet_path(rescaledX, Y_train, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

# COMMAND ----------

# Display results - Lasso and Elastic Net
compare_paths((alphas_lasso, alphas_enet),
              (coefs_lasso, coefs_enet),
              ('Lasso', 'Elastic-Net')
             )

# COMMAND ----------

# Display results - Lasso and Positive Lasso
compare_paths((alphas_lasso, alphas_positive_lasso),
              (coefs_lasso, coefs_positive_lasso),
              ('Lasso', 'positive Lasso')
             )

# COMMAND ----------

# Display results - Elastic Net and Positive Elastic Net
compare_paths((alphas_enet, alphas_positive_enet),
              (coefs_enet, coefs_positive_enet),
              ('Elastic-Net', 'positive Elastic-Net')
             )

# COMMAND ----------

model = Lasso(alpha=0.00145, random_state=0)
model.fit(rescaledX, Y_train)

predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.10 - Keras Regression
# MAGIC Keras is a high-level, deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy.

# COMMAND ----------

# Implementing the R^2 metric

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# COMMAND ----------

# create model
model = Sequential()
model.add(Dense(13, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[coeff_determination])
model.summary()

# Fit the model
early_stopping = EarlyStopping(monitor='coeff_determination', patience=10)
history =  model.fit(X_train, 
                     Y_train, 
                     epochs=150, 
                     batch_size=50, 
                     verbose=0, 
                     validation_data=(X_validation, Y_validation)
                    )

# Evaluate the model
scores = model.evaluate(X_validation, Y_validation, verbose=1)

# COMMAND ----------

hist_df = pd.DataFrame(history.history)
hist_df.tail()

# COMMAND ----------

print(f"{model.metrics_names[0]}: {scores[0]:.4f}")
print(f"{model.metrics_names[1]}: {scores[1]:.4f}")

# COMMAND ----------

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.coeff_determination, label='Training R2', **params_dict)
plt.plot(hist_df.val_coeff_determination, label='Validation R2', **params_dict)
plt.title('R2 for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('R2')
plt.legend()

# COMMAND ----------

# Making predictions
y_pred = model.predict(X_validation)
y_pred[0:10]

# COMMAND ----------

# Using k-fold cross validation
i=1
cvscores0 = []
cvscores1 = []
kfold = KFold(n_splits=10)
for train, validation in kfold.split(X, y):
    # Create model
    model = Sequential()
    model.add(Dense(13, input_dim=X[train].shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[coeff_determination])
    # Fit the model
    model.fit(X[train], 
              y[train], 
              epochs=150, 
              batch_size=20, 
              verbose=0, 
              validation_data=(X[validation], y[validation]),) 
              #callbacks=[early_stopping])
    # evaluate the model
    print('\nResults from #{} run...'.format(i))
    scores = model.evaluate(X[validation], y[validation], verbose=1)
    print("%s: %.4f%%" % (model.metrics_names[0], scores[0]))
    print(f"{model.metrics_names[0]}: {scores[0]:.4f}%")
    print("%s: %.4f%%" % (model.metrics_names[1], scores[1]))
    print(f"{model.metrics_names[1]}: {scores[1]:.4f}%")
    cvscores0.append(scores[0])
    cvscores1.append(scores[1])
    i += 1
# can take >= 5 min

# COMMAND ----------

model.summary()
print(f"{np.mean(cvscores0):.2f}% (+/- {np.std(cvscores0):.4f}%)")
print(f"{np.mean(cvscores1):.2f}% (+/- {np.std(cvscores1):.4f}%)")
