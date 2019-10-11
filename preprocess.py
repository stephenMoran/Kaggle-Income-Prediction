from sklearn import preprocessing
import pandas as pd
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
    return dataset

def encode(train,test, encoder):
    train['train'] = 1
    test['train'] = 0
    combined = pd.concat([train,test])


    #encode
    #combined['Gender'] = encoder.fit_transform(combined['Gender'])
    #combined['Country'] = encoder.fit_transform(combined['Country'])
    combined['Profession'] = encoder.fit_transform(combined['Profession'])
    #combined['University Degree'] = encoder.fit_transform(combined['University Degree'])
    #combined['Hair Color'] = encoder.fit_transform(combined['Hair Color'])


    #columnsToEncode= combined.select_dtypes(include=[object]).columns

    combined= pd.get_dummies(combined, columns=["Gender", "Hair Color","Country", "University Degree"], drop_first=True)
    print(combined.head(n=2))

    train = combined[combined["train"] == 1]
    test = combined[combined["train"] == 0]
    train.drop(["train"], axis=1, inplace=True)
    test.drop(["train"], axis=1, inplace=True)

    train.drop(["Income"], axis=1, inplace=True)
    test.drop(["Income in EUR"], axis=1, inplace=True)

    return train, test

def scale(dataset):
    return dataset
