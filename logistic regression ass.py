#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[50]:


bank=pd.read_csv("bank-full (3).csv",sep=';')
bank 


# In[51]:


bank.info()


# In[53]:


bank.describe()


# In[54]:


bank.shape


# In[55]:


bank.isnull().sum()


# In[56]:


bank.columns


# In[57]:


bank.dtypes


# In[58]:


bank["job"]=bank["job"].astype("category")
bank["marital"]=bank["marital"].astype("category") 
bank["education"]=bank["education"].astype("category")
bank["default"]=bank["default"].astype("category")
bank["housing"]=bank["housing"].astype("category")
bank["loan"]=bank["loan"].astype("category")
bank["contact"]=bank["contact"].astype("category")
bank["month"]=bank["month"].astype("category")
bank["poutcome"]=bank["poutcome"].astype("category")
bank["y"]=bank["y"].astype("category")

bank.dtypes


# In[ ]:





# In[59]:


bank['y'].value_counts() 


# In[60]:


pd.crosstab(bank.job,bank.y).plot(kind='bar')


# In[62]:


pd.crosstab(bank.marital,bank.y).plot(kind='bar')


# In[16]:


pd.crosstab(bank.education,bank.y).plot(kind='bar')


# In[63]:


pd.crosstab(bank.poutcome,bank.y).plot(kind='bar')


# In[64]:


pd.crosstab(bank.month,bank.y).plot(kind='bar')


# In[65]:


sns.barplot(x='age' , y='y' , data=bank)


# In[20]:


sns.barplot(x='balance' , y='y' , data=bank)


# In[66]:


sns.barplot(x='campaign' , y='y' , data=bank)


# In[67]:


sns.barplot(x='previous' , y='y' , data=bank)


# In[68]:


sns.barplot(x='day' , y='y' , data=bank)


# In[24]:


sns.barplot(x='pdays' , y='y' , data=bank)


# In[69]:


sns.barplot(x='duration' , y='y' , data=bank)


# In[70]:


bank['default']=bank['default'].map({'yes':1,'no':0})
bank['loan']=bank['loan'].map({'yes':1,'no':0})
bank['housing']=bank['housing'].map({'yes':1,'no':0})
bank['y']=bank['y'].map({'yes':1,'no':0})
bank=pd.get_dummies(bank,columns=['job','education','marital','poutcome','contact','month'])
bank


# In[71]:


x=pd.concat([bank.iloc[:,0:10],bank.iloc[:,11:]],axis=1)
y=bank.iloc[:,10]


# In[72]:


x


# In[73]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)


# In[74]:


LR=LogisticRegression()


# In[75]:


LR.fit(x_train,y_train)


# In[76]:


y_pred=LR.predict(x_test)
y_pred


# In[77]:


vals = pd.DataFrame({'Actual value':y_test, 'Predicted value':y_pred})


# In[78]:


vals


# In[79]:


# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[80]:


#The model accuracy is calculated by (T.P.+T.N.)/(T.P.+T.N.+F.P.+F.N.)
(11719+366)/(11719+247+1232+366)


# In[81]:


# The model Sensitivity is calculated by (T.P)/(T.P. + F.P)
11719/(11719+1232)


# In[38]:


# The model Specificity is calculated by (T.N)/(T.N. + F.N)
366/(366+247)


# In[82]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[83]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[84]:


pred_prob=LR.predict_proba(x_test)


# In[85]:


prob=pred_prob[:,1]
prob


# In[87]:


fpr,tpr,thersholds=roc_curve(y_test,prob)
plt.plot(fpr,tpr,color="red",label="logistic regression")
plt.xlabel=('false positive rate or [1-true negative rate]')
plt.ylabel=('true positive rate')
plt.plot([0,1],[0,1],'k--')
plt.show


# In[ ]:




