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