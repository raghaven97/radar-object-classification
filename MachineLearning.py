# -*- coding: utf-8 -*-
"""
@author: uids6525
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


__motor_bike = ""
__car_parameters = ""
__truck = ""
__real_pedestrian = ""
__real_pedestrian_on_com = ""
__realpedestrian_stat = ""

__voting_classifier = ""

#Read data from csv 
def read_csv():
    global __motor_bike, __car_parameters, __truck
    global __real_pedestrian, __real_pedestrian_on_com, __realpedestrian_stat
    
    __motor_bike = pd.read_csv('../Radar_Data/motorbike.csv')
    __car_parameters = pd.read_csv('../Radar_Data/car_parameters.csv')
    __truck = pd.read_csv('../Radar_Data/truck.csv')
    __real_pedestrian = pd.read_csv('../Radar_Data/realpedestrian.csv')
    __real_pedestrian_on_com = pd.read_csv('../Radar_Data/realpedestrianoncom.csv')
    __realpedestrian_stat = pd.read_csv('../Radar_Data/realpedestrianstat.csv')
    
# Initial level of processing
def preprocessing():
    
    #global __motor_bike, __car_parameters, __truck
    #global __real_pedestrian, __real_pedestrian_on_com, __realpedestrian_stat 
    
    removeColumns = ['MeasId', ' RectId', 'timestampObject', 'Egovelocity', 'DynamicPropertyObject', 'AccelXObject', 'classIDObject', 'vrelxObject', 'distYObject']
    
    motorbikeProcessed = __motor_bike.drop(removeColumns, axis=1)
    carparametersProcessed = __car_parameters.drop(removeColumns, axis=1)
    #carparametersProcessed = carparametersProcessed.iloc[:50000, :]
    truckProcessed = __truck.drop(removeColumns, axis=1)
    
    removeColumns.extend(['fgridLeft', 'gridConfirmed',  'fgridRight'])
    realpedestrianProcessed = __real_pedestrian.drop(removeColumns, axis=1)
    realpedestrianoncomProcessed = __real_pedestrian_on_com.drop(removeColumns, axis=1)
    realpedestrianstatProcessed = __realpedestrian_stat.drop(removeColumns, axis=1)
    
    
    inputFrames = [motorbikeProcessed, carparametersProcessed, truckProcessed, realpedestrianProcessed, realpedestrianoncomProcessed, realpedestrianstatProcessed]
    return pd.concat(inputFrames, join='outer', join_axes=None, ignore_index=True)

def prepare_for_modelling(preprocessed_dataframe, test_size):
    y = preprocessed_dataframe['Label ']
    #dummies = pd.get_dummies(preprocessed_dataframe['classIDObject'])
    X = preprocessed_dataframe.drop(['Label '], 1)
    #X = X.join(dummies, how='right')
    
    return train_test_split(X, y, test_size = test_size)

def start_fitting(classifier, train, label):
    classifier.fit(train, label)
    
def predict(classifier, test):
    return classifier.predict(test)

def start_voting(classifiers, train, label):
    global __voting_classifier
    __voting_classifier = VotingClassifier(estimators=classifiers, voting='soft')
    __voting_classifier.fit(train, label)

def prediction(test):
    global __voting_classifier
    return __voting_classifier.predict(test)

def predict_probability_matrix(classifier, test):
    return classifier.predict_proba(test)

def create_cross_table(actual, prediction):
    return pd.crosstab(actual, prediction, rownames=['Actual'], colnames=['Predicted'])

def get_accuracy(actual, prediction):
    return accuracy_score(actual, prediction) * 100


def print_accuracy(actual, prediction):
    accuracy = accuracy_score(actual, prediction) * 100
    print("{0:.2f}".format(accuracy) + "%");