import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#data_or = genfromtxt("test_deep_fbl_cpp_feedback_learning_or.dat", delimiter=" ")
#data_xor = genfromtxt("test_deep_fbl_cpp_feedback_learning_xor.dat", delimiter=" ")
#data_and = genfromtxt("test_deep_fbl_cpp_feedback_learning_and.dat", delimiter=" ")
data = genfromtxt("test_deep_fbl_cpp_feedback_learning.dat", delimiter=" ")
err = genfromtxt("test_deep_fbl_cpp_feedback_learning.err", delimiter=" ")
wts = genfromtxt("test_deep_fbl_cpp_feedback_learning.wts", delimiter=" ")

data = data[1:2000,:]
err = err[1:2000,:]
wts = wts[1:2000,:]


numFilters = 3
numInputs = 2
numNeurons = 3


plt.figure(0)
plt.plot(data[:,2], 'k') # state
plt.plot(data[:,3], 'r') # reflex
plt.plot(data[:,9], 'b') # net output
plt.plot(data[:,1], 'y') # unfiltered inputs
plt.plot(data[:,0], 'y') # unfiltered inputs

indx=0
for i in range (2,numNeurons):
    for j in range(numInputs): 
        for k in range(numFilters):
            plt.figure(indx+1)
            plt.plot(10.*wts[:,3+6*indx], 'b') # weight change
            plt.plot(wts[:,5+6*indx], 'y') # filtered inputs
            plt.plot(err[:,4+5*i], 'k') # neuron error
            indx +=1

"""
plt.figure(1)
plt.plot(data[:,2], 'k')
plt.plot(data[:,3], 'r')
plt.plot(data[:,9], 'b')
plt.plot(data[:,5], 'r')

plt.figure(2)
plt.plot(err[:,4], 'b')
plt.figure(3)
plt.plot(err[:,9], 'b')
plt.figure(4)
plt.plot(err[:,14], 'b')

plt.figure(5)
plt.plot(wts[:,1], 'b')
plt.plot(wts[:,5], 'b')
plt.plot(wts[:,9], 'b')
plt.plot(wts[:,13], 'b')
plt.plot(wts[:,17], 'b')
plt.plot(wts[:,21], 'b')
"""

#plt.plot(data[:,6], 'r')
#plt.plot(data[:,7], 'y')

#plt.figure(2)
#plt.plot(data[:,4], 'k')
#plt.plot(data[:,5], 'r')



"""
plt.figure(1)
plt.plot(data_or[:,4], 'k')
plt.plot(data_or[:,5], 'r')
plt.plot(data_or[:,12], 'b')

plt.figure(2)
plt.plot(data_and[:,4], 'k')
plt.plot(data_and[:,5], 'r')
plt.plot(data_and[:,12], 'b')

plt.figure(3)
plt.plot(data_xor[:,4], 'k')
plt.plot(data_xor[:,5], 'r')
plt.plot(data_xor[:,12], 'b')
"""
plt.show()



