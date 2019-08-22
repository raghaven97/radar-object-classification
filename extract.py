# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:53:01 2017

@author: uids6525
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
#from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#from sklearn import utils

motorbike = pd.read_csv('../Radar_Data/motorbike.csv')
carparameters = pd.read_csv('../Radar_Data/car_parameters.csv')
realpedestrian = pd.read_csv('../Radar_Data/realpedestrian.csv')
realpedestrianoncom = pd.read_csv('../Radar_Data/realpedestrianoncom.csv')
realpedestrianstat = pd.read_csv('../Radar_Data/realpedestrianstat.csv')
truck = pd.read_csv('../Radar_Data/truck.csv')

removeColumns = ['fgridLeft', 'gridConfirmed', 'fgridRight', 'DynamicPropertyObject']

motorbikeProcessed = motorbike.drop('DynamicPropertyObject', axis=1)
carparametersProcessed = carparameters.drop('DynamicPropertyObject', axis=1)
truckProcessed = truck.drop('DynamicPropertyObject', axis=1)
realpedestrianProcessed = realpedestrian.drop(removeColumns, axis=1)
realpedestrianoncomProcessed = realpedestrian.drop(removeColumns, axis=1)
realpedestrianstatProcessed = realpedestrian.drop(removeColumns, axis=1)

inputFrames = [motorbikeProcessed, carparametersProcessed, truckProcessed, realpedestrianProcessed, realpedestrianoncomProcessed, realpedestrianstatProcessed]
#,realpedestrian, realpedestrianoncom, realpedestrianstat
totalInput = pd.concat(inputFrames, join='outer', join_axes=None, ignore_index=True)

#enc = OneHotEncoder()

#enc.fit(totalInput['classIDObject'])

#print(enc.transform(totalInput))

print(pd.get_dummies(totalInput['classIDObject']))

print(totalInput.classIDObject.unique())