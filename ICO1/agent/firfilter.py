import numpy as np

class Filterbank:

    def __init__(self, num_filters, min_filterlen = 2, max_filterlen = 10):


        temp_bank = []
        for i in range(0, num_filters-1):
            f = np.array([min_filterlen + i * (max_filterlen - min_filterlen) / num_filters])
            temp_bank.append(f)

        this.bank = np.asarray(temp_bank)






