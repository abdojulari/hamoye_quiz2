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

X = normalized_energydf.drop(columns=['Appliances'])
Y = normalized_energydf['Appliances']
# Y = normalized_energydf.iloc[:, 0].values

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)


# mean absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred).round(2)
print('MAE: ', mae)

# rss 

rss = np.sum(np.square(Y_test -  Y_pred)).round(2)
print('RSS: ', rss)




# mean squared error - also known as RMSE
from sklearn.metrics import mean_squared_error
mse =np.sqrt(mean_squared_error(Y_test, Y_pred)).round(3)
print('MSE', mse)

# R - Squared is coefficient of determination 
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred).round(2)
print('R2:', r2)


# Ridge  (L2 regularisation) 

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(X_train, Y_train)

# Lasso (L1 Regularisation)

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, Y_train)
# Comparing the effects of regularisation 

def get_weights_df(model, feat, col_name):
    #this function returns the weight of every feature
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df

linear_model_weights = get_weights_df(regressor, X_train, 'linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, X_train, 'Ridge_weight')
lasso_weights_df = get_weights_df(lasso_reg, X_train, 'Lasso_Weight')


final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')