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
    #rmse and r2 score
    score = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    print('linear training score ')
    print(score)
    print(rmse)

def corr(f1, f2):
    #correlation plot between features
    dataset.plot(x=f1, y=f2, style='o')
    plt.title(f1p + ' vs ' + f2)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()
