#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing the train and test datasets
mds_df_train=pd.read_csv(r"D:\Puru\PythonDataScience\Mercedes-Benz Greener Manufacturing\train\train.csv")
mds_df_test=pd.read_csv(r"D:\Puru\PythonDataScience\Mercedes-Benz Greener Manufacturing\test\test.csv")


# In[3]:


mds_df_train.shape


# In[4]:


#merging the two datasets to perform data cleaning and encoding categorical variables 
mds_df_train['source']='train'
mds_df_test['source']='test'
mds_df=pd.concat([mds_df_train,mds_df_test], ignore_index=True)


# In[5]:


mds_df.head()


# In[6]:


mds_df.shape


# In[7]:


mds_df.describe(include=object)


# In[8]:


mds_df_train.y.describe()


# In[9]:


#encoding the categorical columns
cat_cols=mds_df.drop('source',axis=1).select_dtypes(include=object).columns
cat_cols


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[11]:


mds_df[cat_cols]=mds_df[cat_cols].apply(le.fit_transform)


# In[12]:


#checking for null values
mds_df.isna().sum().head(15)


# In[13]:


#removing columns based on variance threshold
samp1=mds_df.iloc[::,10:]
var=samp1.var()
samp2=pd.DataFrame(var,columns=['var_value'])
samp2=samp2[samp2['var_value']<0.01]
mds_df.drop(samp2.index,axis=1,inplace=True)


# In[14]:


mds_df.shape


# In[15]:


print('Number of columns dropped with zero variance:', mds_df_train.shape[1]-mds_df.shape[1])


# In[16]:


#seperating the train and test datasets
mds_df_train=mds_df[mds_df['source']=='train']
mds_df_test=mds_df[mds_df['source']=='test']


# In[17]:


#in order to prevent outliers from impacting the model performance, we will try to check and remove outliers from the data
import seaborn as sns
sns.boxplot(mds_df_train.y)


# In[18]:


mds_df_train=mds_df_train.query('70 <= y <= 130')


# In[19]:


mds_df_train.shape


# In[20]:


#checking the data distribution after removing outliers
sns.distplot(mds_df_train.y, bins=10)


# In[21]:


#dropping ID and source columns from train and test datasets as they are not required.
train_df=mds_df_train.drop(['ID','source'],axis=1)
test_df=mds_df_test.drop(['ID','source','y'],axis=1)


# In[22]:


X=train_df.drop('y',axis=1)
Y=train_df.y


# In[23]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X= scaler.fit_transform(X)
#splitting the training datasets into training and testing data
from sklearn.model_selection import train_test_split as tst
xtrain,xtest,ytrain,ytest=tst(X,Y,random_state=42,test_size=0.2)


# In[24]:


#importing and initializing the XGBoostRegressor to train the model using training dataset
from xgboost import XGBRegressor
xgbreg=XGBRegressor(booster='gblinear')


# In[25]:


xgbreg.fit(xtrain,ytrain)


# In[26]:


ypreds=xgbreg.predict(xtest)


# In[27]:


from sklearn.metrics import mean_squared_error as mse
print(np.sqrt(mse(ytest,ypreds)))


# In[28]:


#predicting values for the test datasets
test_df=scaler.fit_transform(test_df)
test_y=xgbreg.predict(test_df)


# In[29]:


final_predictions = pd.DataFrame({'ID':mds_df_test['ID'],'Predictions':test_y})
final_predictions.reset_index(drop=True,inplace=True)
final_predictions['Predictions']=final_predictions['Predictions'].round(2)


# In[30]:


final_predictions


# In[31]:


final_predictions.to_csv(r'D:\Puru\PythonDataScience\Mercedes-Benz Greener Manufacturing\Final Predictions.csv',index=False)


# In[ ]:




