"""
Pure Python implementation of a Back-Propagation Neural Network using the
hyperbolic tangent as the sigmoid squashing function.

Original Author: Neil Schemenauer <nas@arctrix.com>
Modified Author: James Howard <james.w.howard@gmail.com>


Modified to work for function regression and added option to use matplotlib
to display regression networks.

Code is placed in public domain by the two first authors.
==

Modified by Jerzy Karczmarczuk <jerzy.karczmarczuk@unicaen.fr>
* Converted to Python 3 (syntax)
* Uses numpy
* uses Matplotlib.

"""
try:
  from pylab import *      
except:
    print("No Matplotlib")
    from numpy import *

# Make a matrix (NumPy to speed this up)
def makeZ(I, J):
    return zeros((I,J),dtype='double')

    
# calculate a random number where:  a <= rand < b
def rnd(a, b, shp):
    return (b-a)*rand(*shp)+a


class NN(object):
    def __init__(self, ni, nh, no, learnig_rate=0.5, momentum=0.1, no_derivative = True):
        """NN constructor.
        
        ni, nh, no are the number of input, hidden and output nodes.
        regression is used to determine if the Neural network will be trained 
        and used as a classifier or for function regression.
        """
        
        self.N = learnig_rate
        self.M = momentum
        self.nd = no_derivative
        
        #Number of input, hidden and output nodes.
        self.ni = ni  + 1 # +1 for bias node
        self.nh = nh  + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = ones(self.ni)
        self.ah = ones(self.nh)
        self.ao = ones(self.no)
        
        # create weights
        # set them to random values
        self.wi=rnd(-1,1,(self.ni,self.nh))
        self.wo=rnd(-1,1,(self.nh,self.no))
        # last change in weights for momentum   
        self.ci = makeZ(self.ni, self.nh)
        self.co = makeZ(self.nh, self.no)


    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(y):
        if (self.nd) :
            return y
        else :
            return 1.0 - y**2


    def sigmoid(x):
        return tanh(x)


    def step(self, inputs, errors):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            total = 0.0
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i,j]
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total += self.ah[j] * self.wo[j,k]
            self.ao[k] = total
            if not self.regression:
                self.ao[k] = sigmoid(total)

        learn(errors);
	
        return copy(self.ao)  # self.ao[:]



    def learn(self, errors):
        if len(errors) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for hidden
        hidden_deltas = zeros(self.nh)
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += errors[k]*self.wo[j,k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = errors[k]*self.ah[j]
                self.wo[j,k] = self.wo[j,k] + self.N*change + self.M*self.co[j,k]
                self.co[j,k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i,j] = self.wi[i,j] + self.N*change + self.M*self.ci[i,j]
                self.ci[i,j] = change



    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])


def demoDEEPICO():
    # create a network with two input, two hidden, and one output nodes
    net = NN(2, 2, 1)


if __name__ == '__main__':
    demoDEEPICO()
