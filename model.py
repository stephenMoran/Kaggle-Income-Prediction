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
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#xgb
import xgboost as xgb
from xgboost import plot_importance

#own libraries
import preprocess as pp
import visualise as vis

# Making a list of missing value types
missing_values = ['#N/A','0', 'unknown']
dataset = pd.read_csv("./Data/prediction-training/training.csv", na_values = missing_values)
predData = pd.read_csv("./Data/prediction-test/test.csv", na_values = missing_values)

#import pdb; pdb.set_trace()

pd.set_option('display.max_rows', len(dataset))
pd.set_option('display.max_columns', None)

#Clean and encode varibales
dataset = pp.cleanData(dataset)

"""
sns.boxplot(x=dataset['Age'])
plt.show()
"""


predData = pp.cleanData(predData)
encoder = preprocessing.LabelEncoder()
#dataset = pp.removeOutliers(dataset)
dataset, predData = pp.encode(dataset,predData, encoder)
#SPLITTING
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

train = pp.removeOutliers(train)

cols = [col for col in train.columns if col not in ['Instance','Income in EUR']]
X_train = train[cols]

cols = [col for col in test.columns if col not in ['Instance','Income in EUR']]
X_test = test[cols]




#SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#independent variable
y_train = train['Income in EUR']
y_test = test['Income in EUR']


#models
def linear():
    model =  LinearRegression()
    return model

def gradBoost():
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators = 900, max_depth = 5, gamma = 5, colsample_bytree = 0.6)
    return model
#instantiate
regressor =  gradBoost()


#linear fitting
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)
vis.showScore(y_train, y_pred)

y_pred = regressor.predict(X_test)
vis.showScore(y_test, y_pred)


#Prediction

instances = predData['Instance'].values


cols = [col for col in predData.columns if col not in ['Instance','Income']]
predData = predData[cols]

#SCALING
print(predData.shape)
predData = scaler.fit_transform(predData)


pred = regressor.predict(predData)


#EXPORTING
export =  {'Instance': instances, 'Income': pred}
export = pd.DataFrame(export)

export.to_csv('./Data/prediction-submission.csv', index = None)
