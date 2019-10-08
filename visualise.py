def missingValuesSum(dataset):
    #missing values summarised
    print(dataset.isnull().sum()* 100 / len(dataset))
    #total missing values
    print('Total missing values')
    print(dataset.isnull().sum().sum())


    
