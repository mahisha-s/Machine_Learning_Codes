import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Reading the file
df=pd.read_csv("Data files/FuelConsumptionCo2.csv")
print(df.head())
#using matplotlib, plotting graphs
plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS,  color='blue')
plt.show()
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,color="yellow")
plt.show()
#using columns in the csv file whichever is required
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#hist is used for histogram
cdf.hist()
plt.show()
#splitting the data into training and testing set using np library
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
#Linear Regression model starts
from sklearn import linear_model
regr=linear_model.LinearRegression()
#getting the output y from known x using linear regression
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
#y=x0(x) +x1
#x0 is the coefficient and x1 is the intercept
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#plotting the corresponding graph
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color="red")
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.show()
#Evaluating our model
from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_=regr.predict(test_x)
print("Mean absolute error",np.mean(test_y_-test_y))
print("Residual Sum of squares",np.mean((test_y_-test_y)**2))
print("R2 score is ",r2_score(test_y_,test_y))
