# Kaggle-Income-Prediction
Code for Kaggle competition undertaken as part of a final year machine learning module

The code is split into the following files

- **model.py** : This file acts as a main. Here the dataset is loaded in and models are defined before they are fitted and used for prediction. 
- **preprocess.py** : This file deals with missing values, encoding and outlier removal
- **visualise.py** : Here various visualisation techniques are defined and can be used to make sense of the data 

##Methedology

the program follows the sequence of: 
  - Load data 
  - Clean data 
  - Encode data 
  - Split into test and train 
  - Scale data
  - Define model
  - Fit model 
  - Make prediction on train and test 
  - Make prediction on competition data
  
  ##Missing vlaues 
  - Missing values were defined as ['#N/A','0', 'unknown', 'Unknown']
  - Unknown numerical values were replaced with the median 
  - Categorical data was mainly placed into its own 'Unknown' category
  
  
