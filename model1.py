# import required packages
from sklearn.model_selection import train_test_split
import data_prep
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# In[2]:


# splitting the variables(year, brand, body, condition, capacity) and target datasets
X = data_prep.vehicle_dataset.drop(['Price', 'Fuel', 'Mileage', 'Transmission'], axis=1)
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


# R squre Error
error_score = metrics.r2_score(Y_train, train_data_pred)
print("R Squre Error : ", error_score)

# fit() testing data to the linear regression model
lin_reg_model.fit(X_test, Y_test)

# prediction on testing data
test_data_pred = lin_reg_model.predict(X_test)

# setting scatter plot
plt.scatter(Y_test,test_data_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

############## 2. Lasso Regression ################

# In[6]:


# loading the lasso regression model
las_reg_model = Lasso()

# In[7]:


# fit() training data to the lasso regression model
las_reg_model.fit(X_train, Y_train)


#### Model Evaluation ####
# In[8]:


# prediction on training data
train_data_pred = las_reg_model.predict(X_train)


# In[9]:


# R squre Error
error_score = metrics.r2_score(Y_train, train_data_pred)
print("R Squre Error : ", error_score)

# fit() testing data to the lasso regression model
las_reg_model.fit(X_test, Y_test)

# prediction on testing data
test_data_pred = las_reg_model.predict(X_test)

# setting scatter plot
plt.scatter(Y_test,test_data_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
