# Databricks notebook source
# MAGIC %md
# MAGIC # Neural Networks and Deep Learning
# MAGIC ## Intro to Tensor Flow and Keras - optimization
# MAGIC 
# MAGIC Based in this [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning) course. 

# COMMAND ----------

# Setup plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# COMMAND ----------

print(tf.__version__)
try:
    physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(physical_devices)
except:
    print("No GPU")

# COMMAND ----------

from IPython.display import Image 
from IPython.core.display import HTML 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 - The Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 - Cereal
# MAGIC 
# MAGIC The [80 Cereals data set](https://www.kaggle.com/datasets/crawford/80-cereals) contains information about different popular cereal brands and their nutritional contents. 
# MAGIC 
# MAGIC #### Content
# MAGIC Fields in the dataset:
# MAGIC 
# MAGIC - Name: Name of cereal
# MAGIC - mfr: Manufacturer of cereal
# MAGIC   - A = American Home Food Products;
# MAGIC   - G = General Mills
# MAGIC   - K = Kelloggs
# MAGIC   - N = Nabisco
# MAGIC   - P = Post
# MAGIC   - Q = Quaker Oats
# MAGIC   - R = Ralston Purina
# MAGIC - type:
# MAGIC   - cold
# MAGIC   - hot
# MAGIC - calories: calories per serving
# MAGIC - protein: grams of protein
# MAGIC - fat: grams of fat
# MAGIC - sodium: milligrams of sodium
# MAGIC - fiber: grams of dietary fiber
# MAGIC - carbo: grams of complex carbohydrates
# MAGIC - sugars: grams of sugars
# MAGIC - potass: milligrams of potassium
# MAGIC - vitamins: vitamins and minerals - 0, 25, or 100, indicating the typical percentage of FDA recommended
# MAGIC - shelf: display shelf (1, 2, or 3, counting from the floor)
# MAGIC - weight: weight in ounces of one serving
# MAGIC - cups: number of cups in one serving
# MAGIC - rating: a rating of the cereals (Possibly from Consumer Reports?)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 - Red Wine Quality
# MAGIC 
# MAGIC The [Red Wine Quality data set](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) shows properties of different variants of the Portuguese "Vinho Verde" wine. Is consists of physiochemical measurements from about 1600 Portuguese red wines.  Also included is a quality rating for each wine from blind taste-tests. 
# MAGIC These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).
# MAGIC 
# MAGIC #### Content
# MAGIC Input variables (based on physicochemical tests):
# MAGIC - 1 - fixed acidity
# MAGIC - 2 - volatile acidity
# MAGIC - 3 - citric acid
# MAGIC - 4 - residual sugar
# MAGIC - 5 - chlorides
# MAGIC - 6 - free sulfur dioxide
# MAGIC - 7 - total sulfur dioxide
# MAGIC - 8 - density
# MAGIC - 9 - pH
# MAGIC - 10 - sulphates
# MAGIC - 11 - alcohol
# MAGIC Output variable (based on sensory data):
# MAGIC - 12 - quality (score between 0 and 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 - What is Deep Learning?
# MAGIC 
# MAGIC Some of the most impressive advances in artificial intelligence in recent years have been in the field of deep learning. Natural language translation, image recognition, and game playing are all tasks where deep learning models have neared or even exceeded human-level performance.
# MAGIC 
# MAGIC So what is deep learning? Deep learning is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.
# MAGIC 
# MAGIC Through their power and scalability neural networks have become the defining model of deep learning. Neural networks are composed of **neurons**, where each neuron individually performs only a **simple computation**. The power of a neural network comes instead from the **complexity of the connections** these neurons can form.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 - The Linear Unit
# MAGIC Let's begin with the fundamental component of a neural network: the individual neuron.   
# MAGIC As a diagram, a neuron (or unit) with one input looks like:

# COMMAND ----------

Image("./images/mfOlDR6.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 - The Linear Unit as a Model
# MAGIC The input to the neuron is \\(x\\). Its connection to the neuron has a **weight** \\(w\\). Whenever a value flows through a connection, you multiply the value by the connection's weight. For the input x, what reaches the neuron is \\(w \cdot x\\). A neural network "learns" by **modifying its weights**.   
# MAGIC The \\(b\\) is a special kind of weight we call the **bias**. The bias doesn't have any input data associated with it; instead, we put a 1 in the diagram so that the value that reaches the neuron is just \\(b\\) (since \\(1 \cdot b = b\\)). The bias enables the neuron to modify the output independently of its inputs.
# MAGIC 
# MAGIC The \\(y\\) is the value the neuron ultimately **outputs**. To get the output, the neuron sums up all the values it receives through its connections. This neuron's activation is \\(y = w \cdot x + b\\), or as a formula \\(y=w\cdot x+b\\).   
# MAGIC Does the formula \\(y=wx+b\\) look familiar?
# MAGIC It's the equation of a line! It's the slope-intercept equation, where \\(w\\) is the slope and \\(b\\) is the \\(y\\)-intercept. 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Though individual neurons will usually function as part of a **larger network**, it's often useful to start with a **single neuron model** as a baseline. Single neuron models are linear models.
# MAGIC 
# MAGIC Let's think about how this might work on a data set like [80 Cereals](https://www.kaggle.com/crawford/80-cereals). Training a model with 'sugars' (grams of sugars per serving) as input and 'calories' (calories per serving) as output, we might find the bias is \\(b=90\\) and the weight is \\(w=2.5\\). We could estimate the calorie content of a cereal with 5 grams of sugar per serving like this:

# COMMAND ----------

Image("./images/yjsfFvY.png")

# COMMAND ----------

# MAGIC %md
# MAGIC And, checking against our formula, we have calories = \\(2.5 \cdot 5+90=102.5\\), just like we expect.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.2 - Multiple Inputs
# MAGIC 
# MAGIC The 80 Cereals dataset has many more features than just 'sugars'. What if we wanted to expand our model to include things like fiber or protein content? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together.

# COMMAND ----------

Image("./images/vyXSnlZ.png")

# COMMAND ----------

# MAGIC %md
# MAGIC The formula for this neuron would be \\(y=w_0x_0+w_1x_1+w_2x_2+b\\).
# MAGIC 
# MAGIC A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a hyperplane.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 - Linear Units in Keras
# MAGIC 
# MAGIC The easiest way to create a (sequential) model in Keras is through `keras.Sequential`, which creates a neural network as a stack of layers. We can create models like those above using a dense layer (which we'll learn more about later).
# MAGIC 
# MAGIC We could define a linear model accepting three input features ('sugars', 'fiber', and 'protein') and producing a single output ('calories') like so:

# COMMAND ----------

# Create a network with 1 linear unit
model = keras.Sequential([layers.Dense(units=1, input_shape=[3])])

# COMMAND ----------

# MAGIC %md
# MAGIC For our first model we will use the *Red Wine Quality* data set.   
# MAGIC Let's first start by taking a look at the first few rows of the data set. 

# COMMAND ----------

red_wine = pd.read_csv('./data/winequality-red.csv')
red_wine.head()

# COMMAND ----------

# MAGIC %md
# MAGIC How well can we predict a wine's perceived quality from the physiochemical measurements?  
# MAGIC The target is `'quality'`, and the remaining columns are the features.  

# COMMAND ----------

input_shape = [red_wine.shape[1]-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 - Creating a new model for Regression
# MAGIC 
# MAGIC Now define a linear model appropriate for this task. Pay attention to **how many inputs and outputs** the model should have.

# COMMAND ----------

model = keras.Sequential([layers.Dense(units=1, input_shape=input_shape)])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 - Weights
# MAGIC 
# MAGIC Internally, Keras represents the weights of a neural network with **tensors**. Tensors are basically TensorFlow's version of a Numpy array with a few differences that make them better suited to deep learning. One of the most important differences is that tensors are compatible with [GPU](https://www.kaggle.com/docs/efficient-gpu-usage) and [TPU](https://www.kaggle.com/docs/tpu) accelerators. TPUs, in fact, are designed specifically for tensor computations.
# MAGIC 
# MAGIC A model's weights are kept in its `weights` attribute as a list of tensors. 

# COMMAND ----------

w, b = model.weights
print(f"Weights\n{w.numpy()}\n\nBias\n{b.numpy()}")

# COMMAND ----------

# MAGIC %md
# MAGIC Keras represents weights as tensors, but also uses tensors to represent data. When you set the `input_shape` argument, you are telling Keras the dimensions of the array it should expect for each example in the training data. Setting `input_shape=[3]` would create a network accepting vectors of length 3, like `[0.2, 0.4, 0.6]`.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 4.2 - Untrained linear model
# MAGIC  
# MAGIC We will work on *regression* problems, where the goal is to predict some numeric target. Regression problems are like "curve-fitting" problems: we're trying to find a curve that best fits the data. Let's take a look at the "curve" produced by a linear model. 
# MAGIC  
# MAGIC We mentioned that before training a model's weights are set randomly. Run the cell below a few times to see the different lines produced with a random initialization.

# COMMAND ----------

tf.keras.backend.clear_session()

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,)),])

x = tf.linspace(-1.0, 1.0, 100, axis=0)
x = tf.reshape(x, (100,1))
y = model(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 - Layers
# MAGIC 
# MAGIC Neural networks typically organize their neurons into layers.   
# MAGIC When we **collect linear units** having a common set of inputs we get a **dense layer**.

# COMMAND ----------

Image("./images/2MA4iMV.png")

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3.1 - Many Kinds of Layers
# MAGIC     
# MAGIC You could think of each layer in a neural network as performing some kind of relatively simple transformation. Through a deep stack of layers, a neural network can transform its inputs in more and more complex ways. In a well-trained neural network, each layer is a transformation getting us a little bit closer to a solution.
# MAGIC 
# MAGIC A "layer" in Keras is a very general kind of thing. A layer can be, essentially, any kind of data transformation. Many layers, like the convolutional and recurrent layers, transform data through use of neurons and differ primarily in the pattern of connections they form. Others are used for feature engineering or just simple arithmetic. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 4.3.2 - The Activation Function
# MAGIC 
# MAGIC It turns out, however, that two dense layers with nothing in between are no better than a single dense layer by itself.   
# MAGIC Dense layers by themselves can never move us out of the world of lines and planes.   
# MAGIC What we need is something **nonlinear**.    
# MAGIC What we need are **activation functions**.

# COMMAND ----------

Image("./images/OLSUEYT.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Without activation functions, neural networks can only learn linear relationships. In order to fit curves, we'll need to use activation functions.
# MAGIC 
# MAGIC An activation function is simply some function we apply to each of a layer's outputs (its activations).   
# MAGIC The most common is the rectifier function \\(max(0,x)\\). 

# COMMAND ----------

Image("./images/aeIyAlF.png")

# COMMAND ----------

# MAGIC %md
# MAGIC The rectifier function has a graph that's a line with the negative part "rectified" to zero.   
# MAGIC Applying the function to the outputs of a neuron will "bend" the data, moving us away from simple lines.
# MAGIC 
# MAGIC When we attach the rectifier to a linear unit, we get a **rectified linear unit** or **ReLU**. (For this reason, it's common to call the rectifier function the "ReLU function".) Applying a ReLU activation to a linear unit means the output becomes \\(max(0, w \cdot x + b)\\), which we might draw in a diagram like:

# COMMAND ----------

Image("./images/eFry7Yu.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3.3 - Stacking Dense Layers
# MAGIC 
# MAGIC Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations.

# COMMAND ----------

Image("./images/Y5iwFQZ.png")

# COMMAND ----------

# MAGIC %md
# MAGIC A stack of dense layers makes a "fully-connected" network.
# MAGIC 
# MAGIC The layers before the output layer are sometimes called **hidden** since we **never see their outputs** directly.   
# MAGIC Though we haven't shown them in this diagram, each of these neurons would also be receiving a bias (one bias for each neuron).
# MAGIC 
# MAGIC Now, notice that the **final (output) layer is a linear unit** (meaning, no activation function).   
# MAGIC That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value.   
# MAGIC Other tasks (like classification) might **require an activation function** on the output.

# COMMAND ----------

# MAGIC %md
# MAGIC The sequential model we've been using will connect a list of layers in order from first to last: 
# MAGIC - The first layer gets the input
# MAGIC - The last layer produces the output. 
# MAGIC 
# MAGIC This creates the model in the figure above.

# COMMAND ----------

# MAGIC %md
# MAGIC If you by chance need to reset your model waights after running it, this can be done with the following function, which is explained [here](https://stackoverflow.com/questions/40496069/reset-weights-in-keras-layer).

# COMMAND ----------

def reset_model(model):
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
              hasattr(model.layers[ix], 'bias_initializer'):
          weight_initializer = model.layers[ix].kernel_initializer
          bias_initializer = model.layers[ix].bias_initializer

          old_weights, old_biases = model.layers[ix].get_weights()

          model.layers[ix].set_weights([
              weight_initializer(shape=old_weights.shape),
              bias_initializer(shape=old_biases.shape)])
    return model

# COMMAND ----------

model = keras.Sequential([layers.Dense(units=4, activation='relu', input_shape=[2]),
                          layers.Dense(units=3, activation='relu'),
                          layers.Dense(units=1),
                         ])

# COMMAND ----------

model.weights

# COMMAND ----------

# MAGIC %md
# MAGIC **Summary so far**   
# MAGIC We learned how to build fully-connected networks out of stacks of dense layers. When first created, all of the network's weights are set randomly – the network doesn't "know" anything yet. In the next section we're going to see how to train a neural network and how neural networks learn.
# MAGIC 
# MAGIC As with all machine learning tasks, we begin with a set of training data. Each example in the training data consists of some features (the inputs) together with an expected target (the output). Training the network means adjusting its weights in such a way that it can transform the features into the target. In the 80 Cereals dataset, for instance, we want a network that can take each cereal's `'sugar'`, `'fiber'`, and `'protein'` content and produce a prediction for that cereal's `'calories'`. If we can successfully train a network to do that, its weights must represent in some way the relationship between those features and that target as expressed in the training data.
# MAGIC 
# MAGIC In addition to the training data, we need two more things:
# MAGIC 
# MAGIC - A "loss function" that measures how good the network's predictions are.
# MAGIC - An "optimizer" that can tell the network how to change its weights.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 - The Loss Function
# MAGIC 
# MAGIC We've seen how to design an architecture for a network, but we haven't seen how to **tell a network what problem to solve**. This is the job of the **loss function**.   
# MAGIC The loss function measures the disparity between the the target's true value and the value the model predicts.
# MAGIC 
# MAGIC Different problems call for different loss functions. We have been looking at regression problems, where the task is to predict some numerical value – calories in *80 Cereals*, rating in *Red Wine Quality*. Other regression tasks might be predicting the price of a house or the fuel efficiency of a car.
# MAGIC 
# MAGIC A common loss function for regression problems is the **mean absolute error** or **MAE**. For each prediction, MAE measures the disparity from the true target by an absolute difference abs(target - prediction).   
# MAGIC The total MAE loss on a dataset is the mean of all these absolute differences. Hence, it is the average length between the fitted curve and the data points. 

# COMMAND ----------

Image("./images/VDcvkZN.png")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Besides MAE, other loss functions you might see for regression problems are the **mean-squared error** (**MSE**) or the **Huber loss** (both available in Keras).
# MAGIC 
# MAGIC During training, the model will use the loss function as a guide for finding the correct values of its weights (lower loss is better). In other words, the loss function tells the network its objective.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 4.5 - The Optimizer - Stochastic Gradient Descent
# MAGIC 
# MAGIC We've described the problem we want the network to solve, but now we need to say how to solve it. This is the job of the **optimizer**. The optimizer is an algorithm that **adjusts the weights to minimize the loss**.
# MAGIC 
# MAGIC Virtually all of the optimization algorithms used in deep learning belong to a family called **stochastic gradient descent**. They are iterative algorithms that train a network in steps. One step of training goes like this:
# MAGIC 
# MAGIC + Sample some training data and run it through the network to make predictions.
# MAGIC + Measure the loss between the predictions and the true values.
# MAGIC + Finally, adjust the weights in a direction that makes the loss smaller.
# MAGIC 
# MAGIC Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)
# MAGIC 
# MAGIC ![](https://i.imgur.com/rFI1tIk.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 - Training a neural network
# MAGIC 
# MAGIC Each iteration's sample of training data is called a minibatch (or often just "batch"), while a complete round of the training data is called an epoch. The number of epochs you train for is how many times the network will see each training example.
# MAGIC 
# MAGIC The animation shows a linear model being trained with SGD. The pale red dots depict the entire training set, while the solid red dots are the minibatches. Every time SGD sees a new minibatch, it will shift the weights (w the slope and b the y-intercept) toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit. You can see that the loss gets smaller as the weights get closer to their true values.
# MAGIC Learning Rate and Batch Size
# MAGIC 
# MAGIC Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of these shifts is determined by the learning rate. A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.
# MAGIC 
# MAGIC The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)
# MAGIC 
# MAGIC Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 - Adding the Loss and Optimizer
# MAGIC 
# MAGIC After defining a model, you can add a loss function and optimizer with the model's compile method:

# COMMAND ----------

model.compile(optimizer="adam", loss="mae",)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we are able to specify the loss and optimizer with just a string. You can also access these directly through the Keras API -- if you wanted to tune parameters, for instance -- but for us, the defaults will work fine.

# COMMAND ----------

# MAGIC %md
# MAGIC Back to the Red Wine Quality
# MAGIC 
# MAGIC Now we know everything we need to start training deep learning models. So let's see it in action! We'll use the Red Wine Quality dataset.
# MAGIC 
# MAGIC This dataset consists of physiochemical measurements from about 1600 Portuguese red wines. Also included is a quality rating for each wine from blind taste-tests. How well can we predict a wine's perceived quality from these measurements?
# MAGIC 
# MAGIC One thing you might note for now though is that we've rescaled each feature to lie in the interval [0,1]
# MAGIC As we'll discuss more in Lesson 5, neural networks tend to perform best when their inputs are on a common scale.

# COMMAND ----------

red_wine.head()

# COMMAND ----------

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# COMMAND ----------

# MAGIC %md
# MAGIC How many inputs should this network have? We can discover this by looking at the number of columns in the data matrix. Be sure not to include the target ('quality') here -- only the input features.

# COMMAND ----------

len(red_wine.columns) - 1

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC Eleven columns means eleven inputs.
# MAGIC 
# MAGIC We've chosen a three-layer network with over 1500 neurons. This network should be capable of learning fairly complex relationships in the data.

# COMMAND ----------

model = keras.Sequential([layers.Dense(512, activation='relu', input_shape=[11]),
                          layers.Dense(512, activation='relu'),
                          layers.Dense(512, activation='relu'),
                          layers.Dense(1),
                         ])

# COMMAND ----------

# MAGIC %md
# MAGIC Deciding the architecture of your model should be part of a process. Start simple and use the validation loss as your guide.
# MAGIC 
# MAGIC After defining the model, we compile in the optimizer and loss function.

# COMMAND ----------

model.compile(optimizer="adam", loss="mae",)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we're ready to start the training! We've told Keras to feed the optimizer 256 rows of the training data at a time (the batch_size) and to do that 10 times all the way through the dataset (the epochs).

# COMMAND ----------

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=256,
                    epochs=20,
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that Keras will keep you updated on the loss as the model trains.
# MAGIC 
# MAGIC Often, a better way to view the loss though is to plot it. The fit method in fact keeps a record of the loss produced during training in a History object. We'll convert the data to a Pandas dataframe, which makes the plotting easy.

# COMMAND ----------

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df[['loss', 'val_loss']].plot();

# COMMAND ----------

# MAGIC %md
# MAGIC Notice how the loss levels off as the epochs go by. When the loss curve becomes horizontal like that, it means the model has learned all it can and there would be no reason continue for additional epochs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 - Interpreting the Learning Curves
# MAGIC 
# MAGIC You might think about the information in the training data as being of two kinds: signal and noise. The signal is the part that generalizes, the part that can help our model make predictions from new data. The noise is that part that is only true of the training data; the noise is all of the random fluctuation that comes from data in the real-world or all of the incidental, non-informative patterns that can't actually help the model make predictions. The noise is the part might look useful but really isn't.
# MAGIC 
# MAGIC We train a model by choosing weights or parameters that minimize the loss on a training set. You might know, however, that to accurately assess a model's performance, we need to evaluate it on a new set of data, the validation data. (You could see our lesson on model validation in Introduction to Machine Learning for a review.)
# MAGIC 
# MAGIC When we train a model we've been plotting the loss on the training set epoch by epoch. To this we'll add a plot the validation data too. These plots we call the learning curves. To train deep learning models effectively, we need to be able to interpret them.

# COMMAND ----------

Image("./images/tHiVFnM.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the training loss will go down either when the model learns signal or when it learns noise. But the validation loss will go down only when the model learns signal. (Whatever noise the model learned from the training set won't generalize to new data.) So, when a model learns signal both curves go down, but when it learns noise a gap is created in the curves. The size of the gap tells you how much noise the model has learned.
# MAGIC 
# MAGIC Ideally, we would create models that learn all of the signal and none of the noise. This will practically never happen. Instead we make a trade. We can get the model to learn more signal at the cost of learning more noise. So long as the trade is in our favor, the validation loss will continue to decrease. After a certain point, however, the trade can turn against us, the cost exceeds the benefit, and the validation loss begins to rise.

# COMMAND ----------

Image("./images/eUF6mfo.png")

# COMMAND ----------

# MAGIC %md
# MAGIC This trade-off indicates that there can be two problems that occur when training a model: not enough signal or too much noise. Underfitting the training set is when the loss is not as low as it could be because the model hasn't learned enough signal. Overfitting the training set is when the loss is not as low as it could be because the model learned too much noise. The trick to training deep learning models is finding the best balance between the two.
# MAGIC 
# MAGIC We'll look at a couple ways of getting as more signal out of the training data while reducing the amount of noise.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 5.3 - Capacity
# MAGIC 
# MAGIC A model's capacity refers to the size and complexity of the patterns it is able to learn. For neural networks, this will largely be determined by how many neurons it has and how they are connected together. If it appears that your network is underfitting the data, you should try increasing its capacity.
# MAGIC 
# MAGIC You can increase the capacity of a network either by making it wider (more units to existing layers) or by making it deeper (adding more layers). Wider networks have an easier time learning more linear relationships, while deeper networks prefer more nonlinear ones. Which is better just depends on the dataset.

# COMMAND ----------

model = keras.Sequential([layers.Dense(16, activation='relu'),
                          layers.Dense(1),
                         ])

wider = keras.Sequential([layers.Dense(32, activation='relu'),
                          layers.Dense(1),
                         ])

deeper = keras.Sequential([layers.Dense(16, activation='relu'),
                           layers.Dense(16, activation='relu'),
                           layers.Dense(1),
                          ])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 - Early Stopping
# MAGIC 
# MAGIC We mentioned that when a model is too eagerly learning noise, the validation loss may start to increase during training. To prevent this, we can simply stop the training whenever it seems the validation loss isn't decreasing anymore. Interrupting the training this way is called early stopping.

# COMMAND ----------

Image("./images/eP0gppr.png")

# COMMAND ----------

# MAGIC %md
# MAGIC We keep the model where the validation loss is at a minimum.
# MAGIC 
# MAGIC Once we detect that the validation loss is starting to rise again, we can reset the weights back to where the minimum occured. This ensures that the model won't continue to learn noise and overfit the data.
# MAGIC 
# MAGIC Training with early stopping also means we're in less danger of stopping the training too early, before the network has finished learning signal. So besides preventing overfitting from training too long, early stopping can also prevent underfitting from not training long enough. Just set your training epochs to some large number (more than you'll need), and early stopping will take care of the rest.
# MAGIC Adding Early Stopping
# MAGIC 
# MAGIC In Keras, we include early stopping in our training through a callback. A callback is just a function you want run every so ofter while the network trains. The early stopping callback will run after every epoch. (Keras has a variety of useful callbacks pre-defined, but you can define your own, too.)

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(min_delta=0.001, # minimium amount of change to count as an improvement
                               patience=20, # how many epochs to wait before stopping
                               restore_best_weights=True,
                              )

# COMMAND ----------

# MAGIC %md
# MAGIC These parameters say: "If there hasn't been at least an improvement of 0.01 in the validation loss over the previous 5 epochs, then stop the training and keep the best model you found." It can sometimes be hard to tell if the validation loss is rising due to overfitting or just due to noise. The parameters allow us to set some allowances around when to stop.
# MAGIC 
# MAGIC As we'll see in our example, we'll pass this callback to the fit method along with the loss and optimizer.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4.1 - Example: Train a Model with Early Stopping
# MAGIC 
# MAGIC Let's continue developing the model from the example in the last tutorial. We'll increase the capacity of that network but also add an early-stopping callback to prevent overfitting.
# MAGIC 
# MAGIC Let's increase the capacity of the network. We'll go for a fairly large network, but rely on the callback to halt the training once the validation loss shows signs of increasing

# COMMAND ----------

early_stopping = EarlyStopping(min_delta=0.001, # minimium amount of change to count as an improvement
                               patience=20, # how many epochs to wait before stopping
                               restore_best_weights=True,
                              )

model = keras.Sequential([layers.Dense(512, activation='relu', input_shape=[11]),
                          layers.Dense(512, activation='relu'),
                          layers.Dense(512, activation='relu'),
                          layers.Dense(1),
                         ])

model.compile(optimizer='adam',
              loss='mae',
             )

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=256,
                    epochs=500,
                    callbacks=[early_stopping],
                    verbose=0,  # turn off training log
                   )

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

# COMMAND ----------

# MAGIC %md
# MAGIC We'll learn about a two kinds of special layers, not containing any neurons themselves, but that add some functionality that can sometimes benefit a model in various ways. Both are commonly used in modern architectures.
# MAGIC Dropout
# MAGIC 
# MAGIC The first of these is the "dropout layer", which can help correct overfitting.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 - Adding Dropout
# MAGIC 
# MAGIC In Keras, the dropout rate argument rate defines what percentage of the input units to shut off. Put the Dropout layer just before the layer you want the dropout applied to:

# COMMAND ----------

# MAGIC %md
# MAGIC     keras.Sequential([
# MAGIC         # ...
# MAGIC         layer.Dropout(rate=0.3), # apply 30% dropout to the next layer
# MAGIC         layer.Dense(16),
# MAGIC         # ...
# MAGIC     ])

# COMMAND ----------

# MAGIC %md
# MAGIC In the last lesson we talked about how overfitting is caused by the network learning spurious patterns in the training data. To recognize these spurious patterns a network will often rely on very a specific combinations of weight, a kind of "conspiracy" of weights. Being so specific, they tend to be fragile: remove one and the conspiracy falls apart.
# MAGIC 
# MAGIC This is the idea behind dropout. To break up these conspiracies, we randomly drop out some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data. Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust.
# MAGIC 
# MAGIC ![](https://miro.medium.com/max/1048/0*sTulvHwJk7XzmVuW.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, 50% dropout has been added between the two hidden layers.
# MAGIC 
# MAGIC You could also think about dropout as creating a kind of ensemble of networks. The predictions will no longer be made by one big network, but instead by a committee of smaller networks. Individuals in the committee tend to make different kinds of mistakes, but be right at the same time, making the committee as a whole better than any individual. (If you're familiar with random forests as an ensemble of decision trees, it's the same idea.)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 - Batch Normalization
# MAGIC 
# MAGIC The next special layer we'll look at performs "batch normalization" (or "batchnorm"), which can help correct training that is slow or unstable.
# MAGIC 
# MAGIC With neural networks, it's generally a good idea to put all of your data on a common scale, perhaps with something like scikit-learn's StandardScaler or MinMaxScaler. The reason is that SGD will shift the network weights in proportion to how large an activation the data produces. Features that tend to produce activations of very different sizes can make for unstable training behavior.
# MAGIC 
# MAGIC Now, if it's good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better! In fact, we have a special kind of layer that can do this, the batch normalization layer. A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, and then also putting the data on a new scale with two trainable rescaling parameters. Batchnorm, in effect, performs a kind of coordinated rescaling of its inputs.
# MAGIC 
# MAGIC Most often, batchnorm is added as an aid to the optimization process (though it can sometimes also help prediction performance). Models with batchnorm tend to need fewer epochs to complete training. Moreover, batchnorm can also fix various problems that can cause the training to get "stuck". Consider adding batch normalization to your models, especially if you're having trouble during training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.6.1 - Adding Batch Normalization
# MAGIC 
# MAGIC It seems that batch normalization can be used at almost any point in a network. You can put it after a layer...

# COMMAND ----------

layers.Dense(16, activation='relu'),
layers.BatchNormalization(),

# COMMAND ----------

# MAGIC %md
# MAGIC ... or between a layer and its activation function:

# COMMAND ----------

layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),

# COMMAND ----------

# MAGIC %md
# MAGIC And if you add it as the first layer of your network it can act as a kind of adaptive preprocessor, standing in for something like Sci-Kit Learn's StandardScaler.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 5.6.2 - Example: Using Dropout and Batch Normalization
# MAGIC 
# MAGIC Let's continue developing the Red Wine model. Now we'll increase the capacity even more, but add dropout to control overfitting and batch normalization to speed up optimization. This time, we'll also leave off standardizing the data, to demonstrate how batch normalization can stabilize the training.

# COMMAND ----------

del model

# COMMAND ----------

model = keras.Sequential([layers.Dense(1024, activation='relu', input_shape=[11]),
                          layers.Dropout(0.3),
                          layers.BatchNormalization(),
                          layers.Dense(1024, activation='relu'),
                          layers.Dropout(0.3),
                          layers.BatchNormalization(),
                          layers.Dense(1024, activation='relu'),
                          layers.Dropout(0.3),
                          layers.BatchNormalization(),
                          layers.Dense(1),
                         ])

model.compile(optimizer='adam', loss='mae',)

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=256,
                    epochs=100,
                    verbose=0,
                   )

# COMMAND ----------

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(figsize=(8,8));

# COMMAND ----------

# MAGIC %md
# MAGIC You'll typically get better performance if you standardize your data before using it for training. That we were able to use the raw data at all, however, shows how effective batch normalization can be on more difficult datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 - Using NNs for Binary Classification

# COMMAND ----------

# MAGIC %md
# MAGIC So far, we've learned about how neural networks can solve regression problems. Now we're going to apply neural networks to another common machine learning problem: classification. Most everything we've learned up until now still applies. The main difference is in the loss function we use and in what kind of outputs we want the final layer to produce.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Classification into one of two classes is a common machine learning problem. You might want to predict whether or not a customer is likely to make a purchase, whether or not a credit card transaction was fraudulent, whether deep space signals show evidence of a new planet, or a medical test evidence of a disease. These are all binary classification problems.
# MAGIC 
# MAGIC In your raw data, the classes might be represented by strings like "Yes" and "No", or "Dog" and "Cat". Before using this data we'll assign a class label: one class will be 0 and the other will be 1. Assigning numeric labels puts the data in a form a neural network can use.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6.1 - Accuracy and Cross-Entropy
# MAGIC 
# MAGIC Accuracy is one of the many metrics in use for measuring success on a classification problem. Accuracy is the ratio of correct predictions to total predictions: accuracy = number_correct / total. A model that always predicted correctly would have an accuracy score of 1.0. All else being equal, accuracy is a reasonable metric to use whenever the classes in the dataset occur with about the same frequency.
# MAGIC 
# MAGIC The problem with accuracy (and most other classification metrics) is that it can't be used as a loss function. SGD needs a loss function that changes smoothly, but accuracy, being a ratio of counts, changes in "jumps". So, we have to choose a substitute to act as the loss function. This substitute is the *cross-entropy function*.
# MAGIC 
# MAGIC Now, recall that the loss function defines the objective of the network during training. With regression, our goal was to minimize the distance between the expected outcome and the predicted outcome. We chose MAE to measure this distance.
# MAGIC 
# MAGIC For classification, what we want instead is a distance between probabilities, and this is what cross-entropy provides. Cross-entropy is a sort of measure for the distance from one probability distribution to another.

# COMMAND ----------

Image("./images/DwVV9bR.png")

# COMMAND ----------

# MAGIC %md
# MAGIC The idea is that we want our network to predict the correct class with probability 1.0. The further away the predicted probability is from 1.0, the greater will be the cross-entropy loss.
# MAGIC 
# MAGIC The technical reasons we use cross-entropy are a bit subtle, but the main thing to take away from this section is just this: use cross-entropy for a classification loss; other metrics you might care about (like accuracy) will tend to improve along with it.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 - Making Probabilities with the Sigmoid Function
# MAGIC 
# MAGIC The cross-entropy and accuracy functions both require probabilities as inputs, meaning, numbers from 0 to 1. To covert the real-valued outputs produced by a dense layer into probabilities, we attach a new kind of activation function, the sigmoid activation.

# COMMAND ----------

Image("./images/FYbRvJo.png")

# COMMAND ----------

# MAGIC %md
# MAGIC To get the final class prediction, we define a threshold probability. Typically this will be 0.5, so that rounding will give us the correct class: below 0.5 means the class with label 0 and 0.5 or above means the class with label 1. A 0.5 threshold is what Keras uses by default with its accuracy metric.
# MAGIC Example - Binary Classification
# MAGIC 
# MAGIC Now let's try it out!
# MAGIC 
# MAGIC The Ionosphere dataset contains features obtained from radar signals focused on the ionosphere layer of the Earth's atmosphere. The task is to determine whether the signal shows the presence of some object, or just empty air.

# COMMAND ----------

ion = pd.read_csv('./data/ion.csv', index_col=0)
display(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']

# COMMAND ----------

# MAGIC %md
# MAGIC We'll define our model just like we did for the regression tasks, with one exception. In the final layer include a 'sigmoid' activation so that the model will produce class probabilities.

# COMMAND ----------

model = keras.Sequential([layers.Dense(4, activation='relu', input_shape=[33]),
                          layers.Dense(4, activation='relu'),
                          layers.Dense(1, activation='sigmoid'),
                         ])

# COMMAND ----------

# MAGIC %md
# MAGIC Add the cross-entropy loss and accuracy metric to the model with its compile method. For two-class problems, be sure to use 'binary' versions. (Problems with more classes will be slightly different.) The Adam optimizer works great for classification too, so we'll stick with it.

# COMMAND ----------

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy',],
             )

# COMMAND ----------

# MAGIC %md
# MAGIC The model in this particular problem can take quite a few epochs to complete training, so we'll include an early stopping callback for convenience.

# COMMAND ----------

early_stopping = keras.callbacks.EarlyStopping(patience=30,
                                               min_delta=0.001,
                                               restore_best_weights=True,
                                              )

history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=512,
                    epochs=1000,
                    callbacks=[early_stopping],
                    verbose=0, # hide the output because we have so many epochs
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC We'll take a look at the learning curves as always, and also inspect the best values for the loss and accuracy we got on the validation set. (Remember that early stopping will restore the weights to those that got these values.)

# COMMAND ----------

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" + "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
