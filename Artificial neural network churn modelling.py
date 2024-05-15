#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Users/ADMIN/Downloads/archive (25)/Churn_Modelling.csv")


# In[3]:


df


# In[6]:


get_ipython().system('pip install tensorflow')


# In[7]:


import tensorflow as ts                                                                                         


# In[8]:


x=df.iloc[:,3:-1]
y=df.iloc[:,-1]


# In[9]:


x


# In[10]:


y


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


le=LabelEncoder()
x['Gender']=le.fit_transform(x['Gender'])


# In[21]:


print(x)


# In[22]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')


# In[23]:


x=np.array(ct.fit_transform(x))


# In[24]:


print(x)


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[39]:


print(x_train.shape)
x_test.shape


# In[42]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[46]:


ann=ts.keras.models.Sequential()


# In[52]:


ann.add(ts.keras.layers.Dense(units=7,activation='relu'))


# In[53]:


ann.add(ts.keras.layers.Dense(units=7,activation='relu'))


# In[54]:


ann.add(ts.keras.layers.Dense(units=1,activation='sigmoid'))


# In[55]:


ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[56]:


ann.fit(x_train,y_train,batch_size=32,epochs=100)


# In[58]:


df['Geography']


# In[59]:


x


# In[66]:


ann.predict(sc.transform([[1, 0, 0, 600,1,40, 3, 60000, 2, 1, 1, 50000]]))


# In[67]:


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5) #binary classification prediction


# In[69]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:




