import numpy as np
from trace import Trace

class Icolearning:

    def __init__(self, num_inputs, num_filters = 1, learning_rate = 0.0001):

        print ("construcing ICO: num_inputs: ", num_inputs, " num_filters: ", num_filters)
        self.n_inputs = num_inputs
        self.n_filters = num_filters
        self.ntaps = 5
        self.tau = 1
        self.filterBank = []
        self.curr_input = np.zeros(num_inputs)
        self.inputs = np.zeros([num_inputs, self.ntaps])
        self.lastInputs = np.zeros(num_inputs) # needed??
        self.filteredOutputs = np.zeros([num_filters, num_inputs])
        self.weights = np.zeros([num_filters, num_inputs])
        self.lastCorr = np.zeros([num_filters, num_inputs-1])
        self.actualActivity = 0.
        self.diff = np.zeros(num_filters) # we are only keeping the derivative of the reflex
        self.oldOutput = np.zeros(num_filters)
        self.norm = 1.
        self.learningRate = 0. #1e-7

        self.weights[0, 0] = 1 # reflex is always 1

        # put the filter (single filter only) for the reflex first
#        self.filterBank.append([Trace(ntaps=self.ntaps)])

        for i in range(num_filters):
            ntaps = int(float(self.ntaps) / float(i + 1))
            self.filterBank.append(Trace.calCoeffTrace(ntaps, self.tau))

        print self.filterBank


        # Then the filters for the inputs
#        for i in range(1, num_inputs):
#            filterlist = []
#            for j in range(num_filters):
#                filterlist.append(Trace(ntaps=int(float(self.ntaps) / float(j + 1))))
#                print type(filterlist[j])
#                filterlist[j].calCoeffTrace()
#                print filterlist[j].coeffFIR
#            self.filterBank.append(filterlist)

        for i in range(num_filters):
            print ("filterBank: shape=", self.filterBank[i].shape)


    def setCurrInput(self, input):
        self.curr_input = input

    def filter(self):
        # shift
        self.inputs = (np.c_[self.curr_input, self.inputs])[:,:self.ntaps]

        # dot product with filter coefficients
        for i in range(self.n_filters):
            self.filteredOutputs[i,:] = (self.inputs[:,:(self.filterBank[i]).shape[0]]).dot(self.filterBank[i]) / self.norm
            self.diff[i] = self.filteredOutputs[i,0] - self.oldOutput[i]
            self.oldOutput[i] = self.filteredOutputs[i,0]

    def prediction(self, inputs):
        self.setCurrInput(inputs)
        self.filter()

        self.actualActivity = (np.ndarray.flatten(self.filteredOutputs)).dot(np.ndarray.flatten(self.weights))

        for j in range(self.n_filters):
            correl = self.diff[j]*self.filteredOutputs[j, 1:]
            integral = correl - (correl-self.lastCorr[j, :])/2.
            self.weights[j, 1:] = self.weights[j, 1:] + self.learningRate * integral

        return self.actualActivity


    # at the moment, only support setting the entire input block in one operation.
    def setInput(self, f):
        self.inputs = f

