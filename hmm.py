import numpy as np
from preprocess import tokenize_lines

def train_hmm():
    """Takes data as input and returns a HMM modeled using unsupervised training"""
    converged = true
    while not converged:
        e_step() # label the unlabeled data
        m_step() # update the matrices based on labeled data
    pass

def init(num_states, num_emissions):
    """Returns randomly initialized A and O matrices"""
    pass

def e_step():
    """Runs Forward-Backward algorithm to compute marginal probabilities"""
    pass

def m_step():
    """Maximum likelihood estimate of new HMM model parameters"""
    pass

if __name__ == "__main__":
    # tokenize data with preprocess
    data = np.array(tokenize_lines('data/shakespeare.txt'))

    
    # train hmm
    print "Begin training HMM"
    