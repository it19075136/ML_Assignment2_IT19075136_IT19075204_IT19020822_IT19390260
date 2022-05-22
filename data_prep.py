#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# loading data from csv to pandas dataframe
vehicle_dataset = pd.read_csv('./vehicle_data.csv')


# In[4]:


# checking the rows of the dataframe
vehicle_dataset.head()


# In[5]:


# checking the number of rows,columns in dataset
vehicle_dataset.shape


# In[8]:


# checking the dataset columns
vehicle_dataset.info()


# In[6]:


# checking missing value
vehicle_dataset.isnull().sum()


# In[7]:


# checking distribution of categorical data
print(vehicle_dataset.Brand.value_counts())
print(vehicle_dataset.Capacity.value_counts())
print(vehicle_dataset.Year.value_counts())
print(vehicle_dataset.Model.value_counts())
print(vehicle_dataset.Condition.value_counts())
print(vehicle_dataset.Transmission.value_counts())
print(vehicle_dataset.Fuel.value_counts())
print(vehicle_dataset.Location.value_counts())


# In[8]:


# encoding the categorical data
vehicle_dataset.replace({'Fuel':{'Petrol': 0,'Diesel': 1,'Hybrid': 2,'Electric': 3,'CNG': 5,'Other fuel type': 6}},inplace=True)
vehicle_dataset.replace({'Transmission':{'Automatic': 0,'Manual': 1,'Tiptronic': 2,'Other transmission': 3}},inplace=True)
vehicle_dataset.replace({'Condition':{'Used': 0,'Reconditioned': 1,'New': 2}},inplace=True)
vehicle_dataset['Price'] = vehicle_dataset['Price'].str.replace('Rs','')
vehicle_dataset['Price'] = vehicle_dataset['Price'].str.replace(',','').astype(float)
vehicle_dataset['Mileage'] = vehicle_dataset['Mileage'].str.replace('km','')
vehicle_dataset['Mileage'] = vehicle_dataset['Mileage'].str.replace(',','').astype(float)
vehicle_dataset['Capacity'] = vehicle_dataset['Capacity'].str.replace('cc','')
vehicle_dataset['Capacity'] = vehicle_dataset['Capacity'].str.replace(',','').astype(float)


# In[9]:


vehicle_dataset.head()
# %%
