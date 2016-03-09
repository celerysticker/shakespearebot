import numpy as np
from preprocess import tokenize_lines

# Reference: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

def main():
    fname = 'data/shakespeare.txt'
    train_data, obs_dict = load_data(fname)
    
    # pick the number of hidden states
    num_states = 10
    num_obs = len(train_data)
    
    # initialize the A, O matrices
    A, O = init(num_states, num_obs) 
    print A
    print O
    
    
    # train hmm
    print "Begin training HMM"
    #A_trained, O_trained = train_hmm(train_data, num_states, num_obs, A, O)
    print "Complete training HMM"
    

def load_data(fname):
    # tokenize data with preprocess
    data = tokenize_lines(fname)

    num_examples = len(data) # 2155 Shakespeare sonnets
    print "Num of training examples: " + str(num_examples)
    
    obs_dict = dict() #initialize the dictionary of observed words
    num_obs = 0
    train_data = []
    
    # Make a dictionary of all observed tokens (e.g. words, punctuation)
    for i in range(num_examples):
        line = data[i]
        obs_line = []
        for j in range(len(line)):
            word = line[j]
            if word not in obs_dict:
                obs_dict[word] = num_obs
                num_obs += 1
            obs_line.append(obs_dict[word])
        train_data.append(obs_line)
    
    print "Num of distinct obs tokens: " + str(num_obs)
    
    return train_data, obs_dict
    
    
def train_hmm(train_data, num_states, num_obs, A, O):
    """Takes data as input and returns a HMM modeled (A, O matrices)
    using unsupervised training"""
    
    
    converged = 0
    while converged is not 10:
        print converged
        e_step() # label the unlabeled data
        m_step() # update the matrices based on labeled data
        converged += 1
    return 

def init(num_states, num_obs):
    """Returns randomly initialized A and O matrices"""
    A = np.random.uniform(size=(num_states, num_states))
    for row in A:
        row /= np.sum(row)
        
    O = np.random.uniform(size=(num_states, num_obs))
    for row in O:
        row /= np.sum(row)
       
    # make sure that there are no zero values in the initialization
    if np.count_nonzero(A) != num_states**2 or np.count_nonzero(O) != num_states * num_obs:
        print "Initialize again"
        init(num_states, num_obs)
    else:
        return A, O
    

def e_step():
    """Runs Forward-Backward algorithm to compute marginal probabilities"""
    pass

def m_step():
    """Maximum likelihood estimate of new HMM model parameters"""
    pass

if __name__ == "__main__":
    main()
