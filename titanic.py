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
 
label_encoder = LabelEncoder()
train_set.iloc[:, 4] = label_encoder.fit_transform(train_set.iloc[:, 4])
train_set = train_set[pd.notnull(train_set['Embarked'])]
train_set.iloc[:, 11] = label_encoder.fit_transform(train_set.iloc[:, 11])
train_embarked = train_set["Embarked"]
 
label_encoder_1hot = OneHotEncoder(sparse=False)
train_embarked = label_encoder_1hot.fit_transform(train_embarked.values.reshape(-1,1))

test_set.iloc[:, 3] = label_encoder.fit_transform(test_set.iloc[:, 3])
test_set = test_set[pd.notnull(test_set['Embarked'])]
test_set.iloc[:, 10] = label_encoder.fit_transform(test_set.iloc[:, 10])
test_embarked = test_set["Embarked"]
 
test_embarked = label_encoder_1hot.fit_transform(test_embarked.values.reshape(-1,1))

#Removing the original now that its encoded

del train_set["Embarked"]
del test_set["Embarked"]

train_set["EmbarkedS"] = train_embarked[:, 0]
train_set["EmbarkedC"] = train_embarked[:, 1]
#train_set["EmbarkedQ"] = train_embarked[:, 2]

test_set["EmbarkedS"] = test_embarked[:, 0]
test_set["EmbarkedC"] = test_embarked[:, 1]
#test_set["EmbarkedQ"] = test_embarked[:, 2]

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

#Combining the sibling and 

#Seperating into the independent and dependent datasets

X_train = train_set.iloc[:, train_set.columns != 'Survived']
y_train = train_set["Survived"]
X_test = test_set.iloc[:, test_set.columns != 'Survived']

from sklearn.ensemble import RandomForestClassifier

#Fitting the model

reg = RandomForestClassifier()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

#Writing to csv

submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)