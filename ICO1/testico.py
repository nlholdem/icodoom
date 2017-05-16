import numpy as np
from agent.agent import Agent
from agent.firfilter import Filterbank



def main():
    filterbank = Filterbank(10, min_filterlen=1, max_filterlen=20)
    print (filterbank.bank)



if __name__ == '__main__':
    main()


