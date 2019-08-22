# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:39:00 2017

@author: uids6525
"""

import pandas as pd
from sklearn import svm

motorbike = pd.read_csv('../Radar_Data/motorbike.csv')
carparameters = pd.read_csv('../Radar_Data/car_parameters.csv')
realpedestrian = pd.read_csv('../Radar_Data/realpedestrian.csv')
realpedestrianoncom = pd.read_csv('../Radar_Data/realpedestrianoncom.csv')
realpedestrianstat = pd.read_csv('../Radar_Data/realpedestrianstat.csv')
truck = pd.read_csv('../Radar_Data/truck.csv')

svc = svm.SVC()

motorbike_data = motorbike.iloc[:1555, 0:14]
motorbike_label = motorbike.iloc[:1555, 14]

carparameters_data = carparameters.iloc[:, 0:14]
carparameters_label = carparameters.iloc[:, 14]

#realpedestrian_data = realpedestrian.iloc[:, 0:16]
#realpedestrian_label = realpedestrian.iloc[:, 17]

#realpedestrianoncom_data = realpedestrianoncom.iloc[:, 0:16]
#realpedestrianoncom_label = realpedestrianoncom.iloc[:, 17]

#realpedestrianstat_data = realpedestrianstat.iloc[:, 0:16]
#realpedestrianstat_label = realpedestrianstat.iloc[:, 17]

truck_data = truck.iloc[:, 0:14]
truck_label = truck.iloc[:, 14]

inputFrames = [motorbike_data, carparameters_data, truck_data]
x = pd.concat(inputFrames, ignore_index=True)

outputFrames = [motorbike_label, carparameters_label, truck_label]
y = pd.concat(outputFrames, ignore_index=True)

svc.fit(x, y)

inputData = motorbike.iloc[1555:, 0:14]
prediction = svc.predict(inputData)
print(prediction)
