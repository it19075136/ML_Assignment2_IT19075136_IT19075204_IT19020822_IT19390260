#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import required packages
from sklearn.model_selection import train_test_split
import data_prep
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[2]:


# splitting the data and target
X = data_prep.vehicle_dataset.drop(['Price', 'Year', 'Condition', 'Body'], axis=1)
Y = data_prep.vehicle_dataset['Price']


# In[3]:


# print X dataset
print(X)


# In[4]:


# print Y dataset
print(Y)


# In[5]:


# splitting traning and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


#### Model Training ####
############## 1. Linear Regression ################

# In[6]:


# loading the linear regression model
lin_reg_model = LinearRegression()

# In[7]:


# fit() training data to the linear regression model
lin_reg_model.fit(X_train, Y_train)


#### Model Evaluation ####
# In[8]:


# prediction on training data
train_data_pred = lin_reg_model.predict(X_train)


# In[9]:


# R squre Error train data
error_score = metrics.r2_score(Y_train, train_data_pred)
print("R Squre Error : ", error_score)


# In[10]:


#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_train, train_data_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices Vs Predicted Prices")
plt.show()

# In[11]:


# prediction on test data
test_data_pred = lin_reg_model.predict(X_test)


# In[12]:


# R squre error for test data
error_score = metrics.r2_score(Y_test, test_data_pred)
print("R Squre Error : ", error_score)


# In[13]:


#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_test, test_data_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Prices")
plt.show()


############## 2. Lasso Regression ################
# In[14]:


# loading the lasso regression model
lass_reg_model = Lasso()


# In[15]:


# fit() training data to the lasso regression model
lass_reg_model.fit(X_train, Y_train)


#### Model Evaluation ####
# In[16]:


# prediction on training data
train_data_pred = lass_reg_model.predict(X_train)


# In[17]:


# R squre Error train data
error_score = metrics.r2_score(Y_train, train_data_pred)
print("R Squre Error : ", error_score)


# In[18]:


#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_train, train_data_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices Vs Predicted Prices")
plt.show()

# In[19]:


# prediction on test data
test_data_pred = lass_reg_model.predict(X_test)


# In[20]:


# R squre error for test data
error_score = metrics.r2_score(Y_test, test_data_pred)
print("R Squre Error : ", error_score)


# In[21]:


#### Visualize the actual prices vs predicted prices ####
plt.scatter(Y_test, test_data_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Prices")
plt.show()