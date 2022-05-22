#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[3]:


# loading data from csv to pandas dataframe
vehicle_dataset = pd.read_csv('./vehicle_data.csv')


# In[4]:


# checking the rows of the dataframe
vehicle_dataset.head()


# In[5]:


# checking the number of rows,columns in dataset
vehicle_dataset.shape


# In[6]:


# checking the dataset columns
vehicle_dataset.info()


# In[7]:


# checking missing value
vehicle_dataset.isnull().sum()


# In[8]:


# Listed down vehicle dataset columns values
print(vehicle_dataset.columns.tolist())


# In[9]:


# rearrange dataset with remove unnecessary coloums
vehicle_dataset = vehicle_dataset[['Price', 'Brand', 'Year', 'Condition', 'Transmission', 'Body', 'Fuel', 'Capacity', 'Mileage']].copy()
vehicle_dataset.head()


# In[10]:


#rearrange dataset values in Price column
vehicle_price=[]
for item in vehicle_dataset['Price']:
    vehicle_price+=[int(item.replace('Rs ','').replace(',',''))]
vehicle_dataset['Price']=vehicle_price
vehicle_dataset['Price'].unique()


# In[11]:


#rearrange dataset values in Capacity column
vehicle_capacity=[]
for item in vehicle_dataset['Capacity']:
    vehicle_capacity+=[int(item.replace(' cc','').replace(',',''))]
vehicle_dataset['Capacity']=vehicle_capacity
vehicle_dataset['Capacity'].unique()


# In[12]:


#rearrange dataset values in Mileage column
vehicle_mileage=[]
for item in vehicle_dataset['Mileage']:
    vehicle_mileage+=[int(item.replace(' km','').replace(',',''))]
vehicle_dataset['Mileage']=vehicle_mileage
vehicle_dataset['Mileage'].unique()


# In[13]:


# checking missing value from new dataset
vehicle_dataset.isnull().sum()


# In[14]:


# checking distribution of categorical data
print(vehicle_dataset.Brand.value_counts())
print(vehicle_dataset.Condition.value_counts())
print(vehicle_dataset.Transmission.value_counts())
print(vehicle_dataset.Fuel.value_counts())
print(vehicle_dataset.Body.value_counts())

# In[15]


# encode methods def
Numerics = LabelEncoder()
# encoding "Fuel" data
vehicle_dataset['Fuel'] = Numerics.fit_transform(vehicle_dataset['Fuel'])
# encoding "Transmission" data
vehicle_dataset['Transmission'] = Numerics.fit_transform(vehicle_dataset['Transmission'])
# encoding "Condition" data
vehicle_dataset['Condition'] = Numerics.fit_transform(vehicle_dataset['Condition'])
# encoding "Brand" data
vehicle_dataset['Brand'] = Numerics.fit_transform(vehicle_dataset['Brand'])
# encoding "Body" data
vehicle_dataset['Body'] = Numerics.fit_transform(vehicle_dataset['Body'])


# In[16]:


vehicle_dataset.head()