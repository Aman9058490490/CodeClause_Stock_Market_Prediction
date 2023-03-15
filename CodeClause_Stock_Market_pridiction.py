#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install chart_studio')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)


# In[38]:


df=pd.read_csv('GOOG.csv')


# In[39]:


df.head()


# In[62]:


df.info()


# In[63]:


df['date']=pd.to_datetime(df['date'])


# In[64]:


print(f'dataframe contains stock prices between {df.date.min()} {df.date.max()}')
print(f'Total days={(df.date.max() - df.date.min()).days} days')


# In[51]:


df[['open','high','low','close']].plot(kind='box')


# In[65]:


#Setting the layout for our plot
layout =go.Layout(
    title='stock Price of Google',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New,monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
      title='price',
      titlefont=dict(
          family='Courier New, monospace',
          size=18,
          color='#7f7f7f'
      )
    )
)
df_data=[{'x':df['date'],'y':df['close']}]
plot=go.Figure(data=df_data,layout=layout)


# In[66]:


#plot(plot) #plotting offlie:
iplot(plot)


# In[67]:


#Buliding the regression model
from sklearn.model_selection import train_test_split

#for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#for model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[56]:


#split the data into train and test sets
x=np.array(df.index).reshape(-1,1)
y=df['close']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[68]:


#Feature scaling
scaler=StandardScaler().fit(x_train)


# In[69]:


from sklearn.linear_model import LinearRegression


# In[59]:


#creating a linear model
lm=LinearRegression()
lm.fit(x_train,y_train)


# In[70]:


#Plot actual and predicted values for train dataset
trace0=go.Scatter(
    x=x_train.T[0],
    y=y_train,
    mode='markers',
    name='Actual'
)
trace1=go.Scatter(
    x=x_train.T[0],
    y=lm.predict(x_train).T,
    mode='lines',
    name='predicted'
)
df_data=[trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2=go.Figure(data=df_data,layout=layout)


# In[71]:


iplot(plot2)


# In[72]:


#calculate scores for model evaluation
scores=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(y_train,lm.predict(x_train))}\t{r2_score(y_test,lm.predict(x_test))}
{"MSE".ljust(10)}{mse(y_train,lm.predict(x_train))}\t{mse(y_test,lm.predict(x_test))}
'''
print(scores)


# In[ ]:




