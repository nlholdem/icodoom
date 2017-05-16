import numpy as np

class Filterbank:

    def __init__(self, num_filters, min_filterlen = 2, max_filterlen = 10):


        temp_bank = []
        for i in range(0, num_filters-1):
            f = np.full([min_filterlen + i * (max_filterlen - min_filterlen) / num_filters],
                         1. / float(min_filterlen + i * (max_filterlen - min_filterlen) / num_filters) )
            temp_bank.append(f)

#        print temp_bank
        self.bank = np.asarray(temp_bank)






