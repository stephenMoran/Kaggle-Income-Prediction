from sklearn import preprocessing


def cleanData(dataset):
    dataset = missingValues(dataset)
    dataset = encode(dataset)
    return dataset



def missingValues(dataset):
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

def encode(dataset):
    #Encode varibales
    le = preprocessing.LabelEncoder()

    #Encode column lables
    dataset['Gender'] = le.fit_transform(dataset['Gender'])
    dataset['Country'] = le.fit_transform(dataset['Country'])
    dataset['Profession'] = le.fit_transform(dataset['Profession'])
    dataset['University Degree'] = le.fit_transform(dataset['University Degree'])
    dataset['Hair Color'] = le.fit_transform(dataset['Hair Color'])
    return dataset
