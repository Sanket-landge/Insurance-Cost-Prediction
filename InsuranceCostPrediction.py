#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
insurance = pd.read_csv("insurance.csv")
insurance.head()


# In[2]:


# Replacing string values to numbers
insurance['sex'] = insurance['sex'].apply({'male':0,      'female':1}.get) 
insurance['smoker'] = insurance['smoker'].apply({'yes':1, 'no':0}.get)
insurance['region'] = insurance['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[3]:


insurance.head()


# In[4]:


import seaborn as sns
# Correlation betweeen 'charges' and 'age' 
sns.jointplot(x=insurance['age'],y=insurance['charges'])


# In[5]:


# Correlation betweeen 'charges' and 'smoker' 
sns.jointplot(x=insurance['age'],y=insurance['charges'])


# In[6]:


insurance.columns


# In[7]:


# features
X = insurance[['age', 'sex', 'bmi', 'children','smoker','region']]
# predicted variable
y = insurance['charges']


# In[8]:


X.head()


# In[9]:


# importing train_test_split model
from sklearn.model_selection import train_test_split
# splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[10]:


len(X_test) # 402
len(X_train) # 936
len(insurance) # 1338


# In[11]:


# importing the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Fit linear model by passing training dataset
model.fit(X_train,y_train)


# In[12]:


# Predicting the target variable for test datset
predictions = model.predict(X_test)


# In[13]:


predictions[0:5]


# In[14]:


import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[15]:


#Predict charges for new customer : Name- Frank
data = {'age' : 40,
       'sex' : 1,
       'bmi' : 45.50,
       'children' : 4,
       'smoker' : 1,
       'region' : 3}
index = [1]
frank_df = pd.DataFrame(data,index)
frank_df


# In[16]:





prediction_frank = model.predict(frank_df)
print("Medical Insurance cost for Frank is : ",prediction_frank)


# In[ ]:




