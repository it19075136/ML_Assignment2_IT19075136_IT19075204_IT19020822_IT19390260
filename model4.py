#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import required packages
import data_prep
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[2]:


# splitting the data and target
X = data_prep.vehicle_dataset.drop(['Price'], axis=1)
Y = data_prep.vehicle_dataset['Price']


# In[3]:


# print X dataset
print(X)


# In[4]:


# print Y dataset
print(Y)

