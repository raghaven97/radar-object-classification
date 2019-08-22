# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:33:47 2017

@author: uids6525
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#from sklearn import utils

#Read data from csv 
motorbike = pd.read_csv('../Radar_Data/motorbike.csv')
carparameters = pd.read_csv('../Radar_Data/car_parameters.csv')
realpedestrian = pd.read_csv('../Radar_Data/realpedestrian.csv')
realpedestrianoncom = pd.read_csv('../Radar_Data/realpedestrianoncom.csv')
realpedestrianstat = pd.read_csv('../Radar_Data/realpedestrianstat.csv')
truck = pd.read_csv('../Radar_Data/truck.csv')

# Initial level of processing
removeColumns = ['fgridLeft', 'gridConfirmed', 'fgridRight', 'DynamicPropertyObject']

motorbikeProcessed = motorbike.drop('DynamicPropertyObject', axis=1)
carparametersProcessed = carparameters.drop('DynamicPropertyObject', axis=1)
truckProcessed = truck.drop('DynamicPropertyObject', axis=1)
realpedestrianProcessed = realpedestrian.drop(removeColumns, axis=1)
realpedestrianoncomProcessed = realpedestrian.drop(removeColumns, axis=1)
realpedestrianstatProcessed = realpedestrian.drop(removeColumns, axis=1)


inputFrames = [motorbikeProcessed, carparametersProcessed, truckProcessed, realpedestrianProcessed, realpedestrianoncomProcessed, realpedestrianstatProcessed]
totalInput = pd.concat(inputFrames, join='outer', join_axes=None, ignore_index=True)

y = totalInput['Label ']
dummies = pd.get_dummies(totalInput['classIDObject'])
X = totalInput.drop(['Label ', 'classIDObject'], 1)

X = X.join(dummies, how='right')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Classifier instance
#classifier = tree.DecisionTreeClassifier()
#classifier = KNeighborsClassifier(n_neighbors=100)
classifier = RandomForestClassifier()
#classifier = GaussianNB();

# Process of modelling
classifier.fit(X_train, y_train)

# Process of prediction
predictions = classifier.predict(X_test)

# Generates prediction probability matrix
predict_proba = classifier.predict_proba(X_test)

# Generates Actual/Predicted result table
crosstab = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])

#print(list(zip(X, classifier.)))

# Generates overall percentage of accuracy
print(accuracy_score(y_test, predictions))
