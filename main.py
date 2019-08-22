# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:31:50 2017

@author: uids6525
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

motorbike = pd.read_csv('../Radar_Data/motorbike.csv')
carparameters = pd.read_csv('../Radar_Data/car_parameters.csv')
realpedestrian = pd.read_csv('../Radar_Data/realpedestrian.csv')
realpedestrianoncom = pd.read_csv('../Radar_Data/realpedestrianoncom.csv')
realpedestrianstat = pd.read_csv('../Radar_Data/realpedestrianstat.csv')
truck = pd.read_csv('../Radar_Data/truck.csv')

knn = KNeighborsClassifier(n_neighbors=1)

motorbike_data = motorbike.iloc[250:, 0:14]
motorbike_label = motorbike.iloc[250:, 14]

carparameters_data = carparameters.iloc[250:50000, 0:14]
carparameters_label = carparameters.iloc[250:50000, 14]

#realpedestrian_data = realpedestrian.iloc[:, 0:16]
#realpedestrian_label = realpedestrian.iloc[:, 17]

#realpedestrianoncom_data = realpedestrianoncom.iloc[:, 0:16]
#realpedestrianoncom_label = realpedestrianoncom.iloc[:, 17]

#realpedestrianstat_data = realpedestrianstat.iloc[:, 0:16]
#realpedestrianstat_label = realpedestrianstat.iloc[:, 17]

truck_data = truck.iloc[250:, 0:14]
truck_label = truck.iloc[250:, 14]

inputFrames = [motorbike_data, carparameters_data, truck_data]
x = pd.concat(inputFrames, join='outer', join_axes=None, ignore_index=True)

outputFrames = [motorbike_label, carparameters_label, truck_label]
y = pd.concat(outputFrames, join='outer', join_axes=None, ignore_index=True)

knn.fit(x, y)

#inputData = [[5831,	150445,	0.0,	28.80611992,	0.0001,	0,	0.0001,	243.58859249999998,	5970352492,	0.0,	35.2554512,	0,	-13.11749363,	7.294524192999999], 
#            [5847,	154118,	0.0,	4.886171341,	0.0001,	0,	0.0001,	1.946361065,	473554561,	0.0,	24.32402039,	0,	6.770262718,	2.8340890410000004],
#             [5847,	154364,	0.0,	-3.98469162,	0.0001,	0,	0.0001,	2.4012830259999998,	1559398486,	0.0,	22.51312065,	0,	1.364118814,	3.561168194]]


inputBikeData = motorbike.iloc[:250, 0:14]
inputCarData = carparameters.iloc[:250, 0:14]
inputTruckData = truck.iloc[:250, 0:14]

inputDataFrames = [inputBikeData, inputCarData, inputTruckData]
inputData = pd.concat(inputDataFrames, join='outer', join_axes=None, ignore_index=True)

prediction = knn.predict(inputData)
print(prediction)
