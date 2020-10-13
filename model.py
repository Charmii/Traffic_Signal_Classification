import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random

with open("./trafficdata/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./trafficdata/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./trafficdata/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
    
xtrain,ytrain = train['features'],train['labels']
xval,yval = valid['features'],valid['labels']
xtest,ytest = test['features'],test['labels']

i = np.random.randint(1, len(xtrain))
plt.imshow(xtrain[i])
ytrain[i]

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))

axes = axes.ravel() # flaten the 5 x 5 matrix into 25 array

n_training = len(xtrain) # get the length of the training dataset

# Select a random number from 0 to n_training
# create evenly spaces variables 
for i in np.arange(0,W_grid*L_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(xtrain[index])
    axes[i].set_title(ytrain[index],fontsize = 13)
    axes[i].axis('off')
    
    # Select a random number
plt.subplots_adjust(hspace= 0.5)
    # read and display an image with the selected index    
    
from sklearn.utils import shuffle
xtrain, ytrain = shuffle(xtrain, ytrain)

xtraingray = np.sum(xtrain/3,axis = 3, keepdims = True)
xvalidgray = np.sum(xval/3,axis = 3, keepdims = True)
xtestgray = np.sum(xtest/3,axis = 3, keepdims = True)

xtraingraynorm = (xtraingray-128)/128
xvalidgraynorm = (xvalidgray - 128)/128
xtestgraynorm = (xtestgray -128)/128

i = random.randint(1, len(xtraingray))
plt.imshow(xtraingray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(xtrain[i])
plt.figure()
plt.imshow(xtraingraynorm[i].squeeze(), cmap = 'gray')


from tensorflow.keras import datasets, layers, models
CNN = models.Sequential()

CNN.add(layers.Conv2D(6,(5,5),activation = 'relu', input_shape = (32,32,1)))
CNN.add(layers.AveragePooling2D())
CNN.add(layers.Dropout(0.2))
CNN.add(layers.Conv2D(6,(5,5),activation = 'relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())
CNN.add(layers.Dense(120,activation = 'relu'))
CNN.add(layers.Dense(64,activation = 'relu'))
CNN.add(layers.Dense(43,activation = 'softmax'))
CNN.summary()
CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = CNN.fit(xtraingraynorm,
                  ytrain,
                  batch_size = 500,
                  epochs = 5,
                  verbose = 1,
                  validation_data = (xvalidgraynorm,yval))
               
score = CNN.evaluate(xtestgraynorm, ytest)
print('Test Accuracy: {}'.format(score[1]))

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
