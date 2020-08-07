
# coding: utf-8

# In[19]:


#importing dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston #imports the data set for boston model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#charateristics of dataset
boston=load_boston()
print(boston.DESCR)


# In[8]:


#Accessing the data attributes
dataset = boston.data
for index,name in enumerate(boston.feature_names):
    print(index,name)


# In[7]:


#We can pick any of the 13 attibutes.


# In[32]:


#Reshaping data
data=dataset[:,0].reshape(-1,1)
#we piceked crime rate as the deciding attribute.
#reshaping in numpy as (-1,1) is used when there is only one feature. -1 is for python to figure out.


# In[33]:


#shape of data
np.shape(dataset)


# In[34]:


# target values
target = boston.target.reshape(-1,1)


# In[35]:


#shape of target
np.shape(target)


# In[36]:


#graph of the data
plt.scatter(data,target, color='green')


# In[37]:


#more crime , less cost - perfect sense


# In[38]:


#regression
from sklearn.linear_model import LinearRegression
#regression model
reg = LinearRegression()
#fit the model
#no need for separating test and train data. 
reg.fit(data,target)


# In[39]:


#prediction based on the data that was fitted
pred = reg.predict(data)


# In[40]:


#regression line
plt.scatter(data,target, color='green')
plt.plot(data,pred, color='red')#plotting the regression line based on predicted values
plt.xlabel('Crime Rate')
plt.ylabel('Cost of house')
plt.show()

