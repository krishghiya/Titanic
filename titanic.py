# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:22:54 2018

@author: kghiy
"""

#import numpy as np
import pandas as pd

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

#dataset = pd.concat([train_set, test_set])
#train_set.info()
#train_set[['Fare', 'Survived']].groupby(['Fare'],
#        as_index=False).mean().sort_values(by='Survived', ascending=False)

#import seaborn as sns

# =============================================================================
# sns.FacetGrid(train_set, col='Survived').map(plt.hist, 'Pclass', bins=20)

#Map multiple features

#grid = sns.FacetGrid(train_set, col='Survived', row='PClass', height=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();
# =============================================================================

#Categorize Names

for dataset in [train_set, test_set]:
    
    dataset["Title"] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, 
#           "Master": 4, "Rare": 5})
    
#    Add Family - Sum of siblings and parents
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#    Fill NaN values of Embarked
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode())
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

#View frequency

#pd.crosstab(train_set['Title'], train_set['Sex'])

#One hot encoding the "embarked" column for both train and test sets
    
from sklearn.preprocessing import LabelEncoder
 
train_set = pd.get_dummies(train_set, columns=['Embarked', 'Title'])
test_set = pd.get_dummies(test_set, columns=['Embarked', 'Title'])

label_encoder = LabelEncoder()
train_set['Sex'] = label_encoder.fit_transform(train_set['Sex'])
test_set['Sex'] = label_encoder.fit_transform(test_set['Sex'])

#Removing non numeric columns as they cant be compared

#train_set = train_set.select_dtypes(exclude=['object'])
#test_set = test_set.select_dtypes(exclude=['object'])
train_set = train_set.drop(['Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1)
test_set = test_set.drop(['Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1)

#One missing in Fare

test_set = test_set.fillna(test_set.median())

#Grouping Fare
for dataset in [train_set, test_set]:
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 5)
    label = LabelEncoder()
    dataset['FareBin'] = label.fit_transform(dataset['FareBin'])
    
    dataset['AgeBin'] = pd.qcut(dataset['Age'], 4)
    dataset['AgeBin'] = label.fit_transform(dataset['AgeBin'])  

train_set = train_set.drop(['Fare'], axis=1)
test_set = test_set.drop(['Fare'], axis=1)

train_set = train_set.drop(['Age'], axis=1)
test_set = test_set.drop(['Age'], axis=1)

#Combining the sibling and parent columns

#train_set["Family"] = train_set["SibSp"] + train_set["Parch"]

#Seperating into the independent and dependent datasets

X_train = train_set.iloc[:, train_set.columns != 'Survived']
y_train = train_set["Survived"]
X_test = test_set.iloc[:, test_set.columns != 'Survived']

#Standardizing the values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
test_set = scaler.fit_transform(test_set)

from sklearn.ensemble import RandomForestClassifier

#Fitting the model

#reg = RandomForestClassifier(n_estimators=100)
#reg.fit(X_train, y_train)
#y_test = reg.predict(X_test)

#from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier

my_model = XGBClassifier(n_estimators=70)
my_model.fit(X_train, y_train, verbose=False)

# make predictions
y_test = my_model.predict(X_test)

#Writing to csv

submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('submission.csv', index=False)