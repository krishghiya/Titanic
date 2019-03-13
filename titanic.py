# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:22:54 2018

@author: kghiy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

#dataset = pd.concat([train_set, test_set])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#One hot encoding the "embarked" column for both train and test sets
 
train_set = pd.get_dummies(train_set, columns=['Embarked'], drop_first=False)
test_set = pd.get_dummies(test_set, columns=['Embarked'], drop_first=False)

label_encoder = LabelEncoder()
train_set.iloc[:, 4] = label_encoder.fit_transform(train_set.iloc[:, 4])
test_set.iloc[:, 3] = label_encoder.fit_transform(test_set.iloc[:, 3])

#Removing non numeric columns as they cant be compared

train_set = train_set.select_dtypes(exclude=['object'])
test_set = test_set.select_dtypes(exclude=['object'])

#from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import Imputer

#Filling NaN values with mean

imputer = Imputer(strategy="mean")

train_set["Age"] = imputer.fit_transform(train_set["Age"].values.reshape(-1,1))
test_set["Age"] = imputer.fit_transform(test_set["Age"].values.reshape(-1,1))

test_set = test_set.fillna(test_set.median())

#Combining the sibling and parent columns

#train_set["Family"] = train_set["SibSp"] + train_set["Parch"]

#Seperating into the independent and dependent datasets

X_train = train_set.iloc[:, train_set.columns != 'Survived']
y_train = train_set["Survived"]
X_test = test_set.iloc[:, test_set.columns != 'Survived']

from sklearn.ensemble import RandomForestClassifier

#Fitting the model

reg = RandomForestClassifier()
reg.fit(X_train, y_train)
y_test = reg.predict(X_test)

#Writing to csv

submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('submission.csv', index=False)