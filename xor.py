#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
def calculate_layer1(x,w):
        x=x.reshape(1,3)      #reshape x 
        wgt=np.transpose(w)   #find the transpose of w
        layer1 = np.dot(x,wgt) # find the net
        layer1[layer1>0]=1    #use bipolar discrete activation function
        layer1[layer1<=0]=-1
        return layer1


def calculate_layer2(out,w):
        aug_out = np.append(out,[1]) #append 1 to the output of layer1
        layer2=calculate_layer1(aug_out,w)
        return layer



# In[ ]:





# In[ ]:




