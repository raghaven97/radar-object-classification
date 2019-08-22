# -*- coding: utf-8 -*-
"""
@author: uids6525
"""

import MachineLearning as ml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime


ml.read_csv()

preprocessed_dataframe = ml.preprocessing()

X_train, X_test, y_train, y_test = ml.prepare_for_modelling(preprocessed_dataframe, 0.2)

classifiers = [("knn", KNeighborsClassifier(n_neighbors=100)), 
               ("dt", tree.DecisionTreeClassifier()), 
               ("rm", RandomForestClassifier(n_estimators=30)), 
               ("et", ExtraTreesClassifier(n_estimators=30))]

#("svm", svm.SVC(kernel='linear', C=1.0)) // Rejected because of time constraint
start=datetime.now()
ml.start_voting(classifiers, X_train, y_train)
print(datetime.now()-start)

start=datetime.now()
predictions = ml.prediction(X_test)
print(datetime.now()-start)

ml.print_accuracy(y_test, predictions)

print(ml.create_cross_table(y_test, predictions))