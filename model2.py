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

# In[2]

# splitting the data and target
X = data_prep.vehicle_dataset.drop(['Price','Year','Condition','Body','Fuel'], axis=1)
Y = data_prep.vehicle_dataset['Price']



