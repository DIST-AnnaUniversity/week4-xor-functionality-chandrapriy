#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
def calculate_layer1(x,weight1):
        x=x.reshape(1,3)      #reshape x 
        wgt=np.transpose(weight1)   #find the transpose of w
        layer1 = np.dot(x,wgt) # find the net
        layer1[layer1>0]=1    #use bipolar discrete activation function
        layer1[layer1<=0]=-1
        return layer1


def calculate_layer2(out,w):
        aug_out = np.append(out,[1]) #append 1 to the output of layer1
        layer2=calculate_layer1(aug_out,w)
        return layer2

X=np.array([[0,0,1],[0,1,-1],[1,0,-1],[1,1,1],])
print("Input:",x)
print("Input Dimension",X.shape)
weight1=np.array([[-2,1,1/2],[1,-1,1/2],])
print("Dimensions of weight 1",weight1.shape) 
weight2=np.array([[1,1,1],])
print("Dimensions of weight 2",weight2.shape) 
layer1=np.zeros((2,1))
for x in X: 
   
    out_layer1=calculate_layer1(x,weight1)
    print("Output of layer 1",out_layer1)
    out_layer2=calculate_layer2(out_layer1,weight2)
    print("Output of layer2",out_layer2)



# In[ ]:





# In[ ]:




