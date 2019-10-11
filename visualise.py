from sklearn.metrics import r2_score
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def missingValues(dataset):
    #missing values summarised
    print(dataset.isnull().sum()* 100 / len(dataset))
    #total missing values
    print('Total missing values')
    print(dataset.isnull().sum().sum())



def pearson(dataset):
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = dataset.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def showScore(y, y_pred):
    score = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    print('linear training score ')
    print(score)
    print(rmse)
