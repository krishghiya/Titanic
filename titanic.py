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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 
label_encoder = LabelEncoder()
train_set.iloc[:, 4] = label_encoder.fit_transform(train_set.iloc[:, 4])
train_set = train_set[pd.notnull(train_set['Embarked'])]
train_set.iloc[:, 11] = label_encoder.fit_transform(train_set.iloc[:, 11])
 
label_encoder_1hot = OneHotEncoder(categorical_features=[11])
train_set = label_encoder_1hot.fit_transform(train_set).toarray()

from sklearn.metrics import mean_absolute_error

X_train = train_set.iloc[:, train_set.columns != 'Survived']
y_train = train_set["Survived"]
X_test = test_set.iloc[:, test_set.columns != 'Survived']

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
#reg.fit(X_train, y_train)