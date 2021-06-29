#!/usr/bin/env python
# coding: utf-8

# <h1><font color="red">House Price Prediction Model </font></h1>

# # Import of librares

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns


# # Load of Data

# In[2]:


data=pd.read_csv('HousePrices_HalfMil.csv')
data.head()


# # seeing colums(featurs and target of data)

# In[3]:


data.columns


# # data description 
# <p> counting total values of featurs and targests , their mean , maximum value , minimum value, 25% and 75% values </p>

# In[4]:


data.describe()


# # Features selection for model
# <p> Here the Prices are target all other colums are featurs 

# In[5]:


x=data[['Area', 'Garage', 'FirePlace', 'Baths', 'White Marble', 'Black Marble',
       'Indian Marble', 'Floors', 'City', 'Solar', 'Electric', 'Fiber',
       'Glass Doors', 'Swiming Pool', 'Garden']]

x.head()


# In[8]:


y=data[['Prices']]
y.head()


# # Adding a bias vector 

# In[6]:


m=x.shape[0]
a=np.ones((m,1))
x.insert(0,"onces",a)


# # Spliting data in train and test set

# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print('Train set',x_train.shape,y_train.shape)
print('Test set',x_test.shape,y_test.shape)


# # chossing algoritm for model

# In[10]:


model=linear_model.LinearRegression()
train_x=np.asanyarray(x_train)
train_y=np.asanyarray(y_train)


# # changing data type of train_x,y and test_x,y

# In[11]:


train_x=np.asanyarray(x_train)
train_y=np.asanyarray(y_train)
test_x=np.asanyarray(x_test)
test_y=np.asanyarray(y_test)
train_x[0:5]


# In[13]:


model.fit(train_x,train_y)


# # Model evaluation

# <h3> Model accuary </h3>

# In[14]:


y_predict_test=model.predict(test_x)
y_predict_train=model.predict(train_x)
print('accuracy of traning set: {} %'.format(r2_score(y_predict_train, y_train)*100) )


print('accuracy of testing set: {} %'.format( r2_score(y_predict_test, y_test)*100))


# <h3>MSE and MAE Errror calculation</h3>

# In[15]:


print('Mean absolute error of training set: %.2f'% np.mean(np.absolute(y_predict_train - y_train)))
print('Mean absolute error of testing set : %.2f'% np.mean(np.absolute(y_predict_test - y_test)))


# In[17]:


print('Mean Squared Error of training set : %.2f' %mean_squared_error(y_train,y_predict_train))
print('Mean Squared Error of testing set : %.2f' %mean_squared_error(y_test,y_predict_test))

