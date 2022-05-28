# In[1]:
# import required packages

from sklearn.model_selection import train_test_split
import data_prep
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

data_prep.vehicle_dataset.head()

## In[2]:
# %%#spliting the data and target
X = data_prep.vehicle_dataset.drop(['Price','Condition','Transmission','Mileage'],axis=1) 
Y = data_prep.vehicle_dataset['Price']
# %%
## In[3]:
# print X dataset
print(X)
# %%
#print Y dataset
print(Y)
# %%

# splitting traning and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=2)
#### Model Training ####
############## 1. Linear Regression ################
# %%
# loading the linear regression model
linear_reg_model = LinearRegression()

# %%
# fit() training data to the linear regression model
linear_reg_model.fit(X_train,Y_train)
# %%
# prediction on training data
training_data_pre = linear_reg_model.predict(X_train)
# %%
# R squared Error
err_score = metrics.r2_score(Y_train,training_data_pre)
print("R squared Error : ", err_score)
# %%
# Visualize the actual prices and predicted prices
plt.scatter(Y_train,training_data_pre)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
# %%
# fit() testing data to the linear regression model
linear_reg_model.fit(X_test, Y_test)
# %%
# prediction on test data
test_data_pre = linear_reg_model.predict(X_test)
# %%
# R squre error for test data
err_score = metrics.r2_score(Y_test,test_data_pre)
print("R squared Error : ", err_score)
# %%
#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_test,test_data_pre)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
# %%
# loading the lasso regression model
lasso_reg_model = Lasso()
# %%
# fit() training data to the lasso regression model
lasso_reg_model.fit(X_train,Y_train)
# %%
# prediction on training data
training_data_pre = lasso_reg_model.predict(X_train)
# %%
# R squre Error train data
err_score = metrics.r2_score(Y_train,training_data_pre)
print("R Squre Error : ", err_score)
# %%
#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_train,training_data_pre)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
# %%
# fit() testing data to the lasso regression model
lasso_reg_model.fit(X_test,Y_test)
# %%
# prediction on test data
test_data_pre = lasso_reg_model.predict(X_test)
# %%
# R squre error for test data
err_score = metrics.r2_score(Y_test,test_data_pre)
print("R Squre Error : ", err_score)
# %%
#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_test,test_data_pre)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
# %%
