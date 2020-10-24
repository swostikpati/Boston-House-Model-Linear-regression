
# coding: utf-8

# In[79]:


#importing dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston #imports the data set for boston model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


#charateristics of dataset
boston=load_boston()
print(boston.DESCR)


# In[81]:


#Accessing the data attributes
dataset = boston.data
for index,name in enumerate(boston.feature_names):
    print(index,name)


# In[82]:


#We can pick any of the 13 attibutes.


# In[83]:


#Reshaping data
data=dataset[:,5].reshape(-1,1)
#we picked average number of rooms per dwelling as the deciding attribute.
#reshaping in numpy as (-1,1) is used when there is only one feature. -1 is for python to figure out.


# In[84]:


#shape of data
np.shape(dataset)


# In[85]:


# target values
target = boston.target.reshape(-1,1)


# In[86]:


#shape of target
np.shape(target)


# In[87]:


#graph of the data
plt.scatter(data,target, color='green')


# In[88]:


#More the number of rooms per dwelling, More would be the cost of house -- Perfect Sense


# In[89]:


#regression
from sklearn.linear_model import LinearRegression
#regression model
reg = LinearRegression()
#fit the model
#no need for separating test and train data. 
reg.fit(data,target)


# In[90]:


#prediction based on the data that was fitted
pred = reg.predict(data)


# In[91]:


#regression line
plt.scatter(data,target, color='green')
plt.plot(data,pred, color='red')#plotting the regression line based on predicted values
plt.xlabel('Average Number of rooms per dwelling')
plt.ylabel('Cost of house')
plt.show()

