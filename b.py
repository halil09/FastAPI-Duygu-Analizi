#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit

x = streamlit.slider('Select a value')
streamlit.write(x, 'squared is', x * x)


# In[6]:


import numpy as np
yaş = streamlit.selectbox("Yaşınızı seçin: ", np.arange(18, 66, 1))


# In[7]:


age = streamlit.slider("Choose your age: ", min_value=16,   
                       max_value=66, value=35, step=1)


# In[8]:


artists = streamlit.multiselect("Who are your favorite artists?", 
                         ["Michael Jackson", "Elvis Presley",
                         "Eminem", "Billy Joel", "Madonna"])


# In[ ]:




