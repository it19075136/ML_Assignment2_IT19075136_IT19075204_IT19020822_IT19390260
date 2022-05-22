#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# loading data from csv to pandas dataframe
car_dataset = pd.read_csv('./vehicle_data.csv')


# In[4]:


# checking the rows of the dataframe
car_dataset.head()


# In[5]:


# checking the number of rows,columns in dataset
car_dataset.shape


# In[8]:


# checking the dataset columns
car_dataset.info()


# In[6]:


# checking missing value
car_dataset.isnull().sum()


# In[7]:


# checking distribution of categorical data
print(car_dataset.Brand.value_counts())
print(car_dataset.Capacity.value_counts())
print(car_dataset.Year.value_counts())
print(car_dataset.Model.value_counts())
print(car_dataset.Condition.value_counts())
print(car_dataset.Transmission.value_counts())
print(car_dataset.Fuel.value_counts())
print(car_dataset.Location.value_counts())


# In[8]:


# encoding the categorical data
car_dataset.replace({'Fuel':{'Petrol': 0,'Diesel': 1,'Hybrid': 2,'Electric': 3,'CNG': 5,'Other fuel type': 6}},inplace=True)
car_dataset.replace({'Transmission':{'Automatic': 0,'Manual': 1,'Tiptronic': 2,'Other transmission': 3}},inplace=True)
car_dataset.replace({'Condition':{'Used': 0,'Reconditioned': 1,'New': 2}},inplace=True)
car_dataset['Price'] = car_dataset['Price'].str.replace('Rs','')
car_dataset['Price'] = car_dataset['Price'].str.replace(',','').astype(float)
car_dataset['Mileage'] = car_dataset['Mileage'].str.replace('km','')
car_dataset['Mileage'] = car_dataset['Mileage'].str.replace(',','').astype(float)
car_dataset['Capacity'] = car_dataset['Capacity'].str.replace('cc','')
car_dataset['Capacity'] = car_dataset['Capacity'].str.replace(',','').astype(float)


# In[9]:


car_dataset.head()