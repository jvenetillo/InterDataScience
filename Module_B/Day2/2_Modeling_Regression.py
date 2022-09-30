# Databricks notebook source
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest

%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def run_tests():
  unittest.main(argv=[''], verbosity=1, exit=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the data
# MAGIC 
# MAGIC Data [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
# MAGIC Some code imported from [here](https://towardsdatascience.com/predicting-house-prices-with-linear-regression-machine-learning-from-scratch-part-ii-47a0238aeac1)

# COMMAND ----------

!wget https://raw.githubusercontent.com/Data-Science-FMI/ml-from-scratch-2019/master/data/house_prices_train.csv

# COMMAND ----------

df_train = pd.read_csv('house_prices_train.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC # Data exploration

# COMMAND ----------

df_train['SalePrice'].describe()

# COMMAND ----------

sns.distplot(df_train['SalePrice']);

# COMMAND ----------

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), s=32);

# COMMAND ----------

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# COMMAND ----------

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# COMMAND ----------

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# COMMAND ----------

k = 9 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_train[cols].corr(), vmax=.8, square=True);

# COMMAND ----------

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(df_train[cols], size = 4);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Do we have missing data?

# COMMAND ----------

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC # Predicting the sale price
# MAGIC 
# MAGIC ## Preparing the data
# MAGIC 
# MAGIC ### Feature scaling
# MAGIC 
# MAGIC We will do a little preprocessing to our data using the following formula (standardization):
# MAGIC 
# MAGIC $$x'= \frac{x - \mu}{\sigma}$$
# MAGIC 
# MAGIC where $\mu$ is the population mean and $\sigma$ is the standard deviation.
# MAGIC 
# MAGIC ![](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_5/FeatureScaling.jpg)
# MAGIC 
# MAGIC **Source: Andrew Ng**

# COMMAND ----------

x = df_train['GrLivArea']
y = df_train['SalePrice']

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

# COMMAND ----------

x.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC 
# MAGIC ![](https://i.ytimg.com/vi/zPG4NjIkCjc/maxresdefault.jpg)
# MAGIC 
# MAGIC **Source: MyBookSucks**
# MAGIC 
# MAGIC Linear regression models assume that the relationship between a dependent continuous variable $Y$ and one or more explanatory (independent) variables $X$ is linear (that is, a straight line). It’s used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). Linear regression models can be divided into two main types:
# MAGIC 
# MAGIC ### Simple Linear Regression
# MAGIC 
# MAGIC Simple linear regression uses a traditional slope-intercept form, where $a$ and $b$ are the coefficients that we try to “learn” and produce the most accurate predictions. $X$ represents our input data and $Y$ is our prediction.
# MAGIC 
# MAGIC $$Y = bX + a$$
# MAGIC 
# MAGIC ![](https://spss-tutorials.com/img/simple-linear-regression-equation-linear-relation.png)
# MAGIC 
# MAGIC **Source: SPSS tutorials**
# MAGIC 
# MAGIC ### Multivariable Regression
# MAGIC 
# MAGIC A more complex, multi-variable linear equation might look like this, where w represents the coefficients, or weights, our model will try to learn.
# MAGIC 
# MAGIC $$ Y(x_1,x_2,x_3) = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_0$$
# MAGIC 
# MAGIC The variables $x_1, x_2, x_3$ represent the attributes, or distinct pieces of information, we have about each observation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loss function
# MAGIC 
# MAGIC Given our Simple Linear Regression equation:
# MAGIC 
# MAGIC $$Y = bX + a$$
# MAGIC 
# MAGIC We can use the following cost function to find the coefficients:
# MAGIC 
# MAGIC ### Mean Squared Error (MSE) Cost Function
# MAGIC 
# MAGIC The MSE is defined as:
# MAGIC 
# MAGIC $$MSE = J(W) =  \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - h_w(x^{(i)}))^2$$
# MAGIC 
# MAGIC where
# MAGIC 
# MAGIC $$h_w(x) = g(w^Tx)$$
# MAGIC 
# MAGIC The MSE measures how much the average model predictions vary from the correct values. The number is higher when the model is performing "bad" on the training set.
# MAGIC 
# MAGIC The first derivative of MSE is given by:
# MAGIC 
# MAGIC $$MSE' = J'(W) = \frac{2}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})$$
# MAGIC 
# MAGIC 
# MAGIC ### One Half Mean Squared Error (OHMSE)
# MAGIC 
# MAGIC We will apply a small modification to the MSE - multiply by $\frac{1}{2}$ so when we take the derivative, the `2`s cancel out:
# MAGIC 
# MAGIC $$ OHMSE = J(W) =  \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - h_w(x^{(i)}))^2 $$
# MAGIC 
# MAGIC The first derivative of OHMSE is given by:
# MAGIC 
# MAGIC $$OHMSE' = J'(W) = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})$$

# COMMAND ----------

def loss(h, y):
  sq_error = (h - y)**2
  n = len(y)
  return 1.0 / (2*n) * sq_error.sum()

# COMMAND ----------

class TestLoss(unittest.TestCase):

  def test_zero_h_zero_y(self):
    self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([0])), 0)

  def test_one_h_zero_y(self):
    self.assertAlmostEqual(loss(h=np.array([1]), y=np.array([0])), 0.5)

  def test_two_h_zero_y(self):
    self.assertAlmostEqual(loss(h=np.array([2]), y=np.array([0])), 2)
    
  def test_zero_h_one_y(self):
    self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([1])), 0.5)
    
  def test_zero_h_two_y(self):
    self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([2])), 2)

# COMMAND ----------

run_tests()

# COMMAND ----------

class LinearRegression:
  
  def predict(self, X):
    return np.dot(X, self._W)
  
  def _gradient_descent_step(self, X, targets, lr):

    predictions = self.predict(X)
    
    error = predictions - targets
    gradient = np.dot(X.T,  error) / len(X)

    self._W -= lr * gradient
      
  def fit(self, X, y, n_iter=100000, lr=0.01):

    self._W = np.zeros(X.shape[1])

    self._cost_history = []
    self._w_history = [self._W]
    for i in range(n_iter):
      
        prediction = self.predict(X)
        cost = loss(prediction, y)
        
        self._cost_history.append(cost)
        
        self._gradient_descent_step(x, y, lr)
        
        self._w_history.append(self._W.copy())
    return self

# COMMAND ----------

class TestLinearRegression(unittest.TestCase):

    def test_find_coefficients(self):
      clf = LinearRegression()
      clf.fit(x, y, n_iter=2000, lr=0.01)
      np.testing.assert_array_almost_equal(clf._W, np.array([180921.19555322,  56294.90199925]))

# COMMAND ----------

run_tests()

# COMMAND ----------

clf = LinearRegression()
clf.fit(x, y, n_iter=2000, lr=0.01)

# COMMAND ----------

clf._W

# COMMAND ----------

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(clf._cost_history)
plt.show()

# COMMAND ----------

clf._cost_history[-1]

# COMMAND ----------

#Animation

#Set the plot up,
fig = plt.figure()
ax = plt.axes()
plt.title('Sale Price vs Living Area')
plt.xlabel('Living Area in square feet (normalised)')
plt.ylabel('Sale Price ($)')
plt.scatter(x[:,1], y)
line, = ax.plot([], [], lw=2, color='red')
annotation = ax.text(-1, 700000, '')
annotation.set_animated(True)
plt.close()

#Generate the animation data,
def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(-5, 20, 1000)
    y = clf._w_history[i][1]*x + clf._w_history[i][0]
    line.set_data(x, y)
    annotation.set_text('Cost = %.2f e10' % (clf._cost_history[i]/10000000000))
    return line, annotation

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=10, blit=True)

rc('animation', html='jshtml')

anim

# COMMAND ----------

# MAGIC %md
# MAGIC # Multivariable Linear Regression
# MAGIC 
# MAGIC Let's use more of the available data to build a Multivariable Linear Regression model and see whether or not that will improve our OHMSE error:

# COMMAND ----------

x = df_train[['OverallQual', 'GrLivArea', 'GarageCars']]

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

clf = LinearRegression()
clf.fit(x, y, n_iter=2000, lr=0.01)

# COMMAND ----------

clf._W

# COMMAND ----------

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(clf._cost_history)
plt.show()

# COMMAND ----------

clf._cost_history[-1]
