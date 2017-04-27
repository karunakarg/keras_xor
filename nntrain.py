from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy
import os

currentPath = os.path.dirname(os.path.realpath(__file__))+'\\'

# fix random seed for reproducibility
numpy.random.seed(7)

# load  dataset
dataset = numpy.loadtxt(currentPath+"data.txt", delimiter="\t")
print dataset

# split into input (X) and output (Y) variables
X = dataset[:,0:2]
Y = dataset[:,2]
print X
print Y

# create model
model = Sequential() # sequential is keras's regular neural network

# we directly start with 1st hidden layer.
### input_dim is the number of edges coming into each node of this layer, 
### which is same as number of features here, because for first hidden layer, 
### each node gets an incoming edge from all features.
### activation is the activation function we want to use (sigmoid, tanh, relu etc)
### we use tanh, because this problem trains faster using it. Using sigmoid takes ~500 epochs
### Sigmoid takes longer because its range is [0,1]. Tanh is [-1,1].
### For this problem, having negative weights is very helpful. 
### We can learn without -ve weights as well, but it takes a bit longer
model.add(Dense(2, input_dim=2, activation='tanh')) 

# Here we have our output layer with just one node.
# We are training this as a binary classification problem, so one node is enough. 
# We can mark it 0 for first class and 1 for other.
model.add(Dense(1, activation='sigmoid'))

# sgd is Stochastic Gradient Descent, which is the function we'll use to update weights
# lr is learning rate. 0.5 is pretty high learning rate, but for this problem it's a good choice
# you can always experiment with tweaking lr and epochs. Higher the LR, faster it will train.
# But too high of a value will prevent model from converging
sgd = SGD(lr=0.5)

# Compile model
## binary crossentropy is a pretty standard loss function for binary classification
## for optimizer, we'll use previously initialized sgd function
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
## we pass training data X and output labels Y
## epochs is no. of times our net will see complete training data. 
## Here, it'll see all 4 training examples 100 times.
## batch_size :
## instead of individually showing a training example and updating weights,
## we, can show our net a group of training examples and updates weights according to aggregate error.
## this saves time and helps generalize better as our net is not trying to accomodate every single example.
model.fit(X, Y, epochs=100, batch_size=1)

# Print output on input data
print model.predict_proba(X)

# lets have some fun with our model
while True:
	# take csv input from user, e.g. input : 1,1 
	input_values = raw_input("enter values (csv): ")
	# split from comma, this will be a list created from user input, e.g. [1,1]
	input_values = input_values.split(",")
	# convert the input to a numpy array, as keras only understands numpy data types
	input_values = numpy.array(input_values)
	# Neural net takes in a list of input samples and returns a list of predictions for each of them
	# So, we initialize an empty 2D array with 1 row and 2 columns, as we want prediction for 1 sample which has 2 features
	input_vector = numpy.zeros(shape=(1, 2))
	# add our input sample to 2d vector (i.e. input_vector)
	input_vector[0] = input_values
	# print prediction
	print model.predict_proba(input_vector)

