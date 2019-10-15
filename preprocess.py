from sklearn import preprocessing
import pandas as pd
from scipy import stats
import numpy as np


pd.set_option('display.max_columns', None)


def cleanData(dataset):
    #yearOfRecord
    yearMedian =  dataset['Year of Record'].median()
    dataset['Year of Record'].fillna(yearMedian, inplace=True)
    #gender
    dataset['Gender'].fillna('unknown', inplace=True)
    #Age
    ageMedian  = dataset['Age'].median()
    dataset['Age'].fillna(ageMedian, inplace=True)
    #Occupation
    dataset['Profession'].fillna('unknown', inplace=True)
    #UniDegree
    dataset['University Degree'].fillna('No', inplace=True)
    #Hair Color
    dataset['Hair Color'].fillna('Unknown', inplace=True)
    #Wears glasses
    dataset['Wears Glasses'].fillna(0, inplace=True)
    #dataset.drop(["Profession"], axis=1, inplace=True)
    dataset.drop(["Wears Glasses"], axis=1, inplace=True)
    #dataset.drop(["Hair Color"], axis=1, inplace=True)
    #test.drop(["Income in EUR"], axis=1, inplace=True)
    return dataset

def encode(train,test, encoder):
    train['train'] = 1
    test['train'] = 0

    """
    print(len(train))
    q = train["Income in EUR"].quantile(0.99)
    train = train[train["Income in EUR"] < q]
    print(len(train))
    """

    combined = pd.concat([train,test])

    pd.set_option('display.float_format', '{:.2f}'.format)



    #target encode variables
    mean = train["Income in EUR"].median()
    countries = combined.groupby('Country')['Income in EUR'].median()
    combined['Country'] = combined['Country'].map(countries)
    combined['Country'].fillna(mean, inplace=True)

    professions = combined.groupby('Profession')['Income in EUR'].mean()
    combined['Profession'] = combined['Profession'].map(professions)
    combined['Profession'].fillna(mean, inplace=True)

    #encode using label encoder
    #combined['Gender'] = encoder.fit_transform(combined['Gender'])
    #combined['Country'] = encoder.fit_transform(combined['Country'])
    #combined['Profession'] = encoder.fit_transform(combined['Profession'])
    #combined['University Degree'] = encoder.fit_transform(combined['University Degree'])
    #combined['Hair Color'] = encoder.fit_transform(combined['Hair Color'])

    #columnsToEncode= combined.select_dtypes(include=[object]).columns

    #one-hot encode varibales
    combined= pd.get_dummies(combined, columns=["Gender", "Hair Color", "University Degree"], drop_first=True)


    train = combined[combined["train"] == 1]
    test = combined[combined["train"] == 0]
    train.drop(["train"], axis=1, inplace=True)
    test.drop(["train"], axis=1, inplace=True)

    train.drop(["Income"], axis=1, inplace=True)
    test.drop(["Income in EUR"], axis=1, inplace=True)

    return train, test

def removeOutliers(df):
    print(len(df))

    q = df["Body Height [cm]"].quantile(0.99)
    df = df[df["Body Height [cm]"] < q]

    print(len(df))
    return df
