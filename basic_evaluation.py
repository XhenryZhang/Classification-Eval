# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:49:01 2019

@author: xinyi zhang
"""

# seeing how effective our model is
# Classification template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Create predicions for y_train, necessary to do evaluation of the CAP curve
# Could use user entered data for training, or make test set
# X_test: numpy array of test data values to be used in the classifier
# Y_test: numpy array of actual dependent values corresponding to X_test
# classifier: the SKLearn classifier that will be evaluated
def make_cap(X_test, Y_test, classifier):
    y_train_pred = classifier.predict(X_test)
    index_array = []
    model_pred_array = []
    
    for i in range(len(y_train_pred)):
        index_array.append(i)
    
    index_array = np.asarray(index_array)
    
    for index, value in enumerate(y_train_pred):
        model_pred_array.append([index, value])
        
    model_pred_array = np.asarray(model_pred_array)
    sortedResults = np.argsort(model_pred_array[:, -1])
    model_pred_array = model_pred_array[sortedResults, :] # rows are organized based on sorted results
    
    # plotting the random model line
    total_checked = 0
    num_g0 = 0
    num_mod_g0 = 0
    random_model_array = []
    model_pred_array_info = []
    
    for value in y_train:
        if value == 0:
            num_g0 = num_g0 + 1
        total_checked = total_checked + 1
        if total_checked % 3 == 0:
            # make a new entry in numpy array, x is total_checked, y is num_g0
            random_model_array.append([total_checked, num_g0])
    
    total_checked = 0
    
    for value in model_pred_array:
        if y_test[value[0]] == 0:
            num_mod_g0 += 1;
        
        total_checked += 1
        if total_checked % 5 == 0:
            model_pred_array_info.append([total_checked, num_mod_g0])
    
    random_model_array = np.asarray(random_model_array)
    model_pred_array_info = np.asarray(model_pred_array_info)
    
    plt.plot(random_model_array[:, 0], random_model_array[:, 1], color = 'red', label = 'random selection')
    plt.plot(model_pred_array_info[:, 0], model_pred_array_info[:, 1], color = 'blue', label = 'classification model')
    plt.xlabel('Number Surveyed')
    plt.ylabel('Number Accepted')
    plt.legend()
    plt.show()
