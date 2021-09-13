#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import KMeans
import scipy.misc
import matplotlib.pyplot as plt


# In[2]:


pic = scipy.misc.face(gray=True)
plt.figure(figsize=(16,9))
plt.imshow(pic,plt.cm.gray)
plt.show()


# In[4]:


row = pic.shape[0]
column = pic.shape[1]
image = pic.reshape(row*column,1)


# In[5]:


Kmeans = KMeans(n_clusters=5)
Kmeans.fit(image)


# In[7]:


clusters = np.asarray(Kmeans.cluster_centers_)
labels = np.asarray(Kmeans.labels_)
labels = labels.reshape(row,column)


# In[8]:


plt.imsave('compressed_racoon.png',labels)


# In[10]:


image = plt.imread('compressed_racoon.png')
plt.figure(figsize=(16,9))
plt.imshow(image)
plt.show()


# In[ ]:




