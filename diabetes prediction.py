#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[8]:


# The names of the features
print("The names of the features :\n", list(df.columns))


# # #Train and Test

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x=df.iloc[:,df.columns!='Outcome']
y=df.iloc[:,df.columns=='Outcome']


# In[30]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


# In[31]:


xtrain.head()


# In[13]:


ytrain.head()


# ## Data Visualization

# In[14]:


df.hist("Age")


# In[15]:


sns.distplot(df["SkinThickness"])


# In[16]:


sns.set(palette='BrBG')
df.hist(figsize=(20,20));


# # Random Forest Classifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


model=RandomForestClassifier()


# In[19]:


model.fit(xtrain,ytrain.values.ravel())


# In[20]:


predict_output = model.predict(xtest)
print(predict_output)


# In[ ]:





# In[21]:


from sklearn.metrics import accuracy_score


# In[27]:


acc=accuracy_score(predict_output,ytest)
print("The accuracy score for RF:",acc)


# In[ ]:




