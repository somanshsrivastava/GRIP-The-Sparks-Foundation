#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[167]:


#IMPORTING THE DATA AS CSV
url="http://bit.ly/w-data"
df=pd.read_csv(url)


# In[168]:


df.head()


# In[169]:


df.tail()


# In[170]:


df.shape


# In[171]:


df.loc


# In[172]:


df.corr()


# In[173]:


df.isnull()


# In[174]:


df.describe()


# In[175]:


# from this data we can see there is a high correlation between marks scored and number of hours studied


# In[176]:


df.plot(kind='bar')
pyp.title('Relation b/w Hours and Scores')
pyp.xlabel('Hours')
pyp.ylabel('Score')
pyp.show()


# In[177]:


df.plot(kind='area')
pyp.title('Relation b/w Hours and Scores')
pyp.xlabel('Hours')
pyp.ylabel('Score')
pyp.show()


# In[178]:


df.plot(kind='scatter',x='Hours',y='Scores',color='black',figsize=(16,10))
pyp.title('Correlation')
pyp.xlabel('Hours')
pyp.ylabel('Score')
pyp.show()


# In[179]:


#similarly we can visualize this data in many types of graph like area graph, histogram, box plotting,line chart


# In[180]:


#TRAINING


# In[181]:


x=np.asanyarray(df[['Hours']])
y=np.asanyarray(df['Scores'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_,'is your coefficient')
print(regressor.intercept_,'is your interception')


# In[182]:


from sklearn import metrics
from sklearn.metrics import r2_score
predict_y=regressor.predict(x_test)
print('Coefficient of determination is %.2f'%r2_score(predict_y,y_test))


# In[183]:


#the closer the value is to 1 the better the evaluation of the model


# In[184]:


#ACTUAL V/S PREDICTED


# In[185]:


comparison=pd.DataFrame({'Actual':y_test,'Predicted':predict_y})
comparison


# In[186]:


#here we can see that the deviation is give or take 6, so the data that we are working on is quite good and has a high co


# In[193]:


hours=9
predicted_score=regressor.predict([[hours]])
print(f'predicted score={predicted_score[0]}')


# In[ ]:




