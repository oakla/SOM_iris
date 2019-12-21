# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:46:13 2019

@author: oakl
"""
import pandas as pd

data_from_sklearn = False
def get_labels():
    if data_from_sklearn:
        target = genfromtxt('iris.csv',
                            delimiter=',',usecols=(4),dtype=str)
        t = zeros(len(target),dtype=int)
        #**This scheme is for when the target values are strings. Here we need to 
        # convert the targets to ints so that we can index into a marker array**
        t[target == 'setosa'] = 0
        t[target == 'versicolor'] = 1
        t[target == 'virginica'] = 2
    else:
        #** This scheme is for data where the target is just an integer.
        t = iris['target']
    return t
#The first step is to normalize and import the data

from numpy import genfromtxt, array, linalg, zeros, apply_along_axis

if data_from_sklearn:    
    # reading the iris dataset in the csv format
    #**** If you already have iris.csv on your harddrive ****
    # (downloaded from http://aima.cs.berkeley.edu/data/iris.csc)
    data = genfromtxt('iris.csv', delimiter=',',usecols=(0,1,2,3))
else:
    ##**** if you want to use iris.csv from scikitlearn ****
    from sklearn import datasets
    iris = datasets.load_iris() 
    #iris is a bunch, which is like a dictionary but you can access it like an object
    print(iris.keys())
    print(type(iris.data))
    # iris.data is a numpy array
    data = iris.data

# now we need to normalize each input (pattern) in the data
data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)

print(data.shape)

# Now we can start the training process
from minisom import MiniSom
### initialization and training ###
som = MiniSom(7,7,4,sigma=1.0,learning_rate=0.5)
som.random_weights_init(data)
print("Training...")
som.train_random(data,100) # training with 100 iterations
print("\n...ready!")


'''Printing'''
# Let's visualize the results of training.
# We shall call the printing function from Ross's code

# The following code plots the "average distance map" and marks the neuron
# that each data point falls into
# ....

from matplotlib.pyplot import plot,axis,show,pcolor,colorbar,bone
bone() # set the colormap to "bone"
pcolor(som.distance_map().T) # distance map as background
colorbar()
# loading the labels
t = get_labels()

# use different colors and markers for each label
markers = ['o','s','D']
colors = ['r','g','b']
for cnt,xx in enumerate(data):
    w = som.winner(xx) # obviously returns the winner for this datapoint
    # the following code places a marker on the winner position(neuron) for DP xx
    plot(w[0]+.5,w[1]+0.5, markers[t[cnt]],markerfacecolor='None',
         markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
axis([0,som.get_weights().shape[0],0,som.get_weights().shape[1]])

show()
        
