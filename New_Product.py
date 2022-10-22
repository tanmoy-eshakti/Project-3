#!/usr/bin/env python
# coding: utf-8

# In[73]:


import streamlit as st
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn import preprocessing
from sklearn.svm import SVR
import pickle


data = pd.read_csv('New Item Offline.csv')
filename = 'finalized_model.sav'

#Pageview per user
Viewperuser = 242222765/17903179
data['conversion'] = data['Orders']/(data['Users']*Viewperuser)
data.fillna('', inplace=True)
#df = pd.DataFrame(data,columns=['Productid','ProductStyle','Color','ColorGroup','Fabric','Occassion','Silhouette','Season','Pattern','FabricType'])
le = preprocessing.LabelEncoder()
data['ProductStyle']= le.fit_transform(data['ProductStyle'])
X1 = le.inverse_transform(data['ProductStyle'])
data['ColorGroup']= le.fit_transform(data['ColorGroup'])
X2 = le.inverse_transform(data['ColorGroup'])
data['Occassion']= le.fit_transform(data['Occassion'])
X5 = le.inverse_transform(data['Occassion'])
data['Silhouette']= le.fit_transform(data['Silhouette'])
X6 = le.inverse_transform(data['Silhouette'])
data['Season']= le.fit_transform(data['Season'])
X4 = le.inverse_transform(data['Season'])
data['Pattern']= le.fit_transform(data['Pattern'])
X3 = le.inverse_transform(data['Pattern'])

#le.fit(['Productid','ProductStyle','Color','ColorGroup','Fabric','Occassion','Silhouette','Season','Pattern','FabricType'])
#LabelEncoder()


X = data[['ProductStyle','ColorGroup','Pattern','Season','Occassion','Silhouette']]
y = data['conversion']

ProductStyle = st.selectbox('Choose your Type Of Clothing',('Denim Dress','Denim Jacket','Denim Shirts','Denim Skirt','Dress','Jacket','Jeans','Jumpsuit','Pant','Skirt','Top'))
ColorGroup = st.selectbox('Choose your Type Of Color Type',('Beige','Black','Blue','Brown','Gold','Gray','Green','Indigo','Multi-colored','Orange','Pink','Purple','Red','Violet','White','Yellow'))
Pattern = st.selectbox('Choose your Type Of Print',('Animal Print','Colorblock','Floral','Graphic','Plaid','Stripes','Whimsical'))
Season = st.selectbox('You are most likely to use clothing during?',('Bridal','Fall','Winter','Fall-Winter','Fall/Winter','Neutral','Perennial','Spring','Spring/Summer','Summer','Wedding'))
Occassion = st.selectbox('Choose your most likely occassion',('Party','Special Occasion','Vacation','Work'))
Silhouette = st.selectbox('If Dress is your type of clothing, what is your preferred Silhouette?',('A-Line','Fit-and-flare','Maxi','Sheath','Shift','Shirtdress','Wrap'))

X_test = {
  "ProductStyle": np.where(np.unique(X1)==ProductStyle),
  "ColorGroup": np.where(np.unique(X2)==ColorGroup),
  "Pattern": np.where(np.unique(X3)==Pattern),
  "Season": np.where(np.unique(X4)==Season),
  "Occassion": np.where(np.unique(X5)==Occassion),
  "Silhouette": np.where(np.unique(X6)==Silhouette)}

# Load the pickled model
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(pd.DataFrame(X_test))
probability = predictions * 100
st.write('Likelihood % for the new product: '+ str(round(probability[0],2)) + '%')




# In[ ]:





# In[ ]:





# In[ ]:




