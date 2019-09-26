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

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting RFC to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# entropy criterion: higher disorder of particles, high entropy
# algorithm aims to lower entropy in child nodes from parent nodes, this creates information gain

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max()) # set limit to scatter plt
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Create predicions for y_train, necessary to do evaluation of the CAP curve
# could use user entered data for training, or make test set
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
    if y_train[value[0]] == 0:
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
