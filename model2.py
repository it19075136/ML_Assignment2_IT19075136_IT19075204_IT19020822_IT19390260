#In[1]

#import required packages
from numpy import matrix
from sklearn.model_selection import train_test_split
import data_prep
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# In[2]:

# splitting the data and target
X = data_prep.vehicle_dataset.drop(['Price','Year','Body','Fuel'], axis=1)
Y = data_prep.vehicle_dataset['Price']

# In[3]:

# print X dataset
print(X)

# In[4]:

# print Y dataset
print(Y)

# In[5]:

# splitting traininig and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

#### Model Training ####
############## 1. Linear Regression ################

# In[6]:

# loading the linear regression model
linear_regression_model = LinearRegression()

# In[7]:

# fit() training data to the linear regression model
linear_regression_model.fit(X_train, Y_train)

#### Model Evaluation ####

# In[8]:

# prediction on training data
training_data_prediction = linear_regression_model.predict(X_train)

# In[9]

# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared  Error : ", error_score)

# In[10]

# Visualize the actual price and predicted prices


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

# In[11]

# prediction on testing data
test_data_prediction = linear_regression_model.predict(X_test)

# In[12]

# R squared Error
error_score_test = metrics.r2_score(Y_test , test_data_prediction)
print("R squared Error Test : ", error_score_test) 

# In[13]

# Visualize the actual prices and predicted prices

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

############## 2. Lasso Regression ################

# In[14]

# Using the lasso regression model

lasso_regression_model = Lasso()

# In[15]

# fit() training data to the lasso regression model
lasso_regression_model.fit(X_train, Y_train)

#### Model Evaluation ####

# In[16]

# R square Error
error_score_lasso = metrics.r2_score(Y_train, training_data_prediction)
print("R Square Error : ",error_score_lasso)

# In[17]

# fit() testing data to the lasso regression model
lasso_regression_model.fit(X_test, Y_test)

# In[18]

# prediction on testing data lasso model
testing_data_prediction = lasso_regression_model.predict(X_test)

# In[19]

# setting scatter plot
plt.scatter(Y_test,testing_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predcited Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()