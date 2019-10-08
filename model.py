import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#xgb
import xgboost as xgb
from xgboost import plot_importance
import dataClean as clean

# Making a list of missing value types
missing_values = ['#N/A','0', 'unknown']
dataset = pd.read_csv("./Data/prediction-training/training.csv", na_values = missing_values)

#import pdb; pdb.set_trace()

pd.set_option('display.max_rows', len(dataset))
pd.set_option('display.max_columns', None)


dataset = clean.cleanData(dataset)

#Encode varibales
le = preprocessing.LabelEncoder()

#Encode column lables
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Country'] = le.fit_transform(dataset['Country'])
dataset['Profession'] = le.fit_transform(dataset['Profession'])
dataset['University Degree'] = le.fit_transform(dataset['University Degree'])
dataset['Hair Color'] = le.fit_transform(dataset['Hair Color'])

"""
columnsToEncode=dataset.select_dtypes(include=[object]).columns
dataset = pd.get_dummies(dataset, columns=columnsToEncode, drop_first=True)
"""

#Feature Selections
cols = [col for col in dataset.columns if col not in ['Instance','Income in EUR']]
X = dataset[cols]
y = dataset['Income in EUR']

"""
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
"""

#SPLITTING
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

cols = [col for col in train.columns if col not in ['Instance','Income in EUR']]
X_train = train[cols]

cols = [col for col in test.columns if col not in ['Instance','Income in EUR']]
X_test = test[cols]

#SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#dependent variable
y_train = train['Income in EUR']
y_test = test['Income in EUR']


#TRAINING

def linearModel():
    model =  LinearRegression()
    return model

print(X_train[:10])
#instantiate
regressor =  xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators = 800, max_depth = 4) #objective="reg:squarederror", random_state=42
print(regressor)


#FITTING DATA
#linear fitting
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_train)
score = r2_score(y_train, y_pred)
rmse = sqrt(mean_squared_error(y_train, y_pred))
print('linear training score ')
print(score)
print(rmse)


y_pred = regressor.predict(X_test)
score = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('linear testing score ')
print(score)
print(rmse)

#To retrieve the intercept:
#print(regressor.intercept_)
#For retrieving the slope:
#print(regressor.coef_)

#PREDICTION
missing_values = ['#N/A','0', 'unknown']
test = pd.read_csv("./Data/prediction-test/test.csv", na_values = missing_values)

#replacement of values
#yearOfRecord
yearMedian =  test['Year of Record'].median()
test['Year of Record'].fillna(yearMedian, inplace=True)
#gender
test['Gender'].fillna('unknown', inplace=True)
#Age
ageMedian  = test['Age'].median()
test['Age'].fillna(ageMedian, inplace=True)
#Occupation
test['Profession'].fillna('unknown', inplace=True)
#UniDegree
test['University Degree'].fillna('No', inplace=True)
#Hair Color
test['Hair Color'].fillna('Unknown', inplace=True)
#Wears Glasses
test['Wears Glasses'].fillna(0, inplace=True)

#Encode column lables
test['Gender'] = le.fit_transform(test['Gender'])
test['Country'] = le.fit_transform(test['Country'])
test['Profession'] = le.fit_transform(test['Profession'])
test['University Degree'] = le.fit_transform(test['University Degree'])
test['Hair Color'] = le.fit_transform(test['Hair Color'])



"""
columnsToEncode=test.select_dtypes(include=[object]).columns
test = pd.get_dummies(dataset, columns=columnsToEncode, drop_first=True)
"""

instances = test['Instance'].values
print(instances[0])

cols = [col for col in test.columns if col not in ['Instance','Income']]
test = test[cols]

#SCALING
test = scaler.fit_transform(test)

#PREDICTION
pred = regressor.predict(test)


#EXPORTING
export =  {'Instance': instances, 'Income': pred}
export = pd.DataFrame(export)

export.to_csv('./Data/prediction-submission.csv', index = None)
