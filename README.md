# Kaggle-Income-Prediction
Code for Kaggle competition undertaken as part of a final year machine learning module

The code is split into the following files

- **model.py** : This file acts as a main. Here the dataset is loaded in and models are defined before they are fitted and used for prediction. 
- **preprocess.py** : This file deals with missing values, encoding and outlier removal
- **visualise.py** : Here various visualisation techniques are defined and can be used to make sense of the data 

## Methedology

The program follows the sequence of: 
  - Load data 
  - Clean data 
  - Encode data 
  - Split into test and train 
  - Scale data
  - Define model
  - Fit model 
  - Make prediction on train and test 
  - Make prediction on competition data
  
## Missing values and feature selection 
  - Missing values were defined as ['#N/A','0', 'unknown', 'Unknown']
  - Unknown numerical values were replaced with the median 
  - Categorical data was mainly placed into its own 'Unknown' category
  - From looking at a correlation matrix, wears glasses was deemed an irrelevant feature and therefore was dropped
  
## Encoding values and Scaling
#### Encoding 
Two different methiods of encoding were used for categorical features: 
  - One-hot encoding: used for 'Gender', 'Hair Color' and 'University Degree'
  - Target encoding: used for 'Country' and 'Profession'
Found one-hot encoding unsuitable for Country and Profession, where there were a large number of labels 

To acccount for the fact that values would appear in the competition data that were not in the training data. The encoding fucntion takes both datasets as parametres and encodes everything before splitting the data back into training and competition data

#### Scaling 
Standard sclaer was used to fit all the data 

## Outliers
Through visualisation I noticed noise and outliers in the data. However, I found that only altering 'Bodyt Height[cm]' gave a performace enchancement

I removed everyting outside the 0.99 quantile 

## Model 
After experimenting with many models such as linear, ridge, lasso and SVM, I foudn the best performance with XGBRegressor 

Using the following parameteres

```
 xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators = 900, max_depth = 5, gamma = 5, colsample_bytree = 0.7)
```

I was able to achieve a RMSE under 61,000 on the public leaderboard 

