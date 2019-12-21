# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:46:13 2019

@author: oakl
"""
import pandas as pd
from numpy import genfromtxt,array, linalg, zeros, apply_along_axis

data_from_sklearn = False #alternatively you have a loval copy of the data

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

if data_from_sklearn:    
    # reading the iris dataset in the csv format
    #**** If you already have iris.csv on your harddrive ****
    # (downloaded from http://aima.cs.berkeley.edu/data/iris.csc)
    attributes_df = pd.read_csv('iris.csv', delimiter=',',usecols=(0,1,2,3))
else:
    ##**** if you want to use iris.csv from scikitlearn ****
    from sklearn import datasets
    iris = datasets.load_iris() 
    #iris is a bunch, which is like a dictionary but you can access it like an object
    print(iris.keys())
    print(type(iris.data))
    # iris.data is a numpy array
    attributes_df = pd.DataFrame(iris.data)

import seaborn as sn
import matplotlib.pyplot as plt

def boxplot(df, ymin=None, ymax=None):
    # create new figure
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    # set axis labels and scale
    ax.tick_params(axis='x', labelrotation=90)
    ax.set(ylabel='Attribute value')
    if not ymin==None and not ymax==None:
        plt.ylim([ymin, ymax])
    
    # plot boxplot using seaborn
    sn.boxplot(data=df, orient='v', ax=ax)
    
boxplot(attributes_df)  

import numpy as np

# create empty dataframe with headings for each attribute
attribute_scaling_df = pd.DataFrame(columns=attributes_df.columns[:])

# set scaling for each attribute to 'linear' by default
a = list()
for i in range(0, len(attribute_scaling_df.columns)):
    a.append('linear')
attribute_scaling_df.loc[0] = a

# manually change scaling for attributes
attribute_scaling_df['attribute_1'][0] = 'log'
attribute_scaling_df['attribute_24'][0] = 'log'
attribute_scaling_df['attribute_25'][0] = 'log'
attribute_scaling_df['attribute_26'][0] = 'log'
#attribute_scaling_df['attribute_34'][0] = 'log'
#attribute_scaling_df['attribute_35'][0] = 'log'
#attribute_scaling_df['attribute_36'][0] = 'log'
#attribute_scaling_df['attribute_37'][0] = 'log'
print(attribute_scaling_df)
# def self_organising_maps(attributes_df, size=32):
#     # define matrix of attributes and values
#     X = attributes_df.iloc[:, 2:len(attributes_df.columns)]

#     # run self-organising maps algorithm
#     # create an object
#     som = MiniSom(x=size, y=size, input_len=len(attributes_df.columns) - 2, sigma=1.5*np.sqrt(size), learning_rate=0.5)
#     ### maybe sigma=5*np.sqrt(size) and learning_rate=1
#     # initialize the weights and train
#     som.random_weights_init(X.values)
#     som.train_random(X.values, num_iteration=2*len(attributes_df))
#     #som.train_batch(X.values, num_iteration=len(attributes_df)) # this gives more predictable results
    
#     print 'There are '+str(len(attributes_df))+' events assigned to '+str(size*size)+' neurons.'
    
#     return som


#     # plot boxplot using seaborn
#     sn.boxplot(data=attributes_df.drop(columns=['start_time', 'stop_time']), orient='v', ax=ax)

# # now we need to normalize each input (pattern) in the data
# data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)

# # Now we can start the training process
# from minisom import MiniSom
# ### initialization and training ###
# som = MiniSom(7,7,4,sigma=1.0,learning_rate=0.5)
# som.random_weights_init(data)
# print("Training...")
# som.train_random(data,100) # training with 100 iterations
# print("\n...ready!")


# '''Printing'''
# # Let's visualize the results of training.
# # We shall call the printing function from Ross's code
# from SOM_Printing import distance_plotter
# df = pd.DataFrame(data)
# distance_plotter(df,som)  
