import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #plotting

energydf = pd.read_csv('/Users/Oj/dataset/energydata_complete.csv')
energydf.head()
energydf = energydf.drop(columns= ['date','lights'])


# feature scaling and data normalization 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_energydf = pd.DataFrame(scaler.fit_transform(energydf), columns=energydf.columns)

# group dataset into target value and independent variables 
normalized_energydf.head()

X = normalized_energydf[['T2']]
Y = normalized_energydf[['T6']]
# Y = normalized_energydf.iloc[:, 0].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#X_train = X_train.reshape(-1,1)
#Y_train = Y_train.reshape(-1,1)
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)



# R - Squared is coefficient of determination 
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred).round(2)
print('R2:', r2)


