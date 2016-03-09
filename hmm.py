import numpy as np
from preprocess import tokenize_lines

# Reference: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

def main():
    fname = 'data/shakespeare.txt'
    train_data, obs_dict = load_data(fname)
    
    # pick the number of hidden states
    num_states = 10
    num_obs = len(obs_dict)
    
    # initialize the A, O matrices
    A, O = init(num_states, num_obs) 
    print "Dimensions of A: {0}".format(A.shape)
    print "Dimension of O: {0}".format(O.shape)
    
    # train hmm
    print "..........Begin training HMM.........."
    A, O = train_hmm(train_data, num_states, num_obs, A, O)
    print "..........Complete training HMM.........."
    
    A_str = latex_matrix(A) # write hmm to file
    O_str = latex_matrix(O)
    with open('h1.txt', 'w') as f:
        f.write(A_str)
        f.write(O_str)
    
def latex_matrix(matrix):
    matrix_str = ''
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_str += str("{0:.3f}".format(matrix[i][j])) + '\t'
        matrix_str = matrix_str[:-2] + '\n'
    return matrix_str

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
    converged = False
    
    iteration = 0
    while iteration is not 10: # fixed number of iterations to be 10
        print "Iteration " + str(iteration)
        A, O = EM_step(train_data, num_states, num_obs, A, O)
        iteration += 1
    return A, O

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
    

def EM_step(train_data, num_states, num_obs, A, O):
    """Runs Forward-Backward algorithm to compute marginal probabilities """
    num_train = len(train_data)
        
    obs = train_data[0] # train on the first line
    len_ = len(obs)
    # Forward subalgorithm (computed recursively)
    alpha = [[[0.] for i in range(num_states)] for i in range(len_)]
    alpha[0] = [(1. / num_states) * O[i][obs[0]] for i in range(num_states)]
    
    for length in range(1, len_):
        for state in range(num_states):
            p_obs = O[state][obs[length]] # probability of observing data in  our given 'state'
            p_trans = 0  # probability of transitioning to 'state'
            
            for prev_state in range(num_states): # iterate through all possible prev states
                p_trans += alpha[length - 1][prev_state] * A[prev_state][state]
            alpha[length][state] = p_trans * p_obs # update probability
        #row_sum = sum(alpha[length])
        #alpha[length] = [a/row_sum for a in alpha[length]] #renormalize alpha
    

    # Backward subalgorithm 
    beta = [[[0.] for i in range(num_states)] for i in range(len_)] 
    beta[len_ - 1] = [1 for i in range(num_states)]
    
    for length in range(len_ - 2, -1, -1):
        for state in range(num_states):
            p_end = 0
            for next_state in range(num_states):
                p_end += beta[length + 1][next_state] * A[state][next_state] * \
                        O[next_state][obs[length + 1]]
            beta[length][state] = p_end
        #row_sum = sum(beta[length])
        #beta[length] = [b/row_sum for b in beta[length]] #renormalize beta
        
    # Update values
    gamma = [[[0.] for i in range(num_states)] for i in range(len_)] 
    
    for length in range(len_):
        for state in range(num_states):
            state_sum = 0
            for s in range(num_states):
                state_sum += alpha[length][s] * beta[length][s]
            gamma[length][state] = alpha[length][state] * beta[length][state] / state_sum
    
    epsilon = [[[[0.] for i in range(num_states)] for i in range(num_states)] for i in range(len_)]

    for length in range(len_ - 1):
        for j in range(num_states):
            for i in range(num_states):
                epsilon[length][j][i] = alpha[length][i] * A[i][j] * beta[length + 1][j] * \
                                        O[j][obs[length + 1]] / sum(alpha[len_ - 1])
                
    # M-STEP: numerator and denominator sum over every training sequence
    # in the training set
                
    A = np.zeros(shape=(num_states, num_states))
    
    for i in range(num_states):
        for j in range(num_states):
            numerator = 0
            denominator = 0
            for length in range(len_ - 1):
                numerator += epsilon[length][j][i]
                denominator += gamma[length][i]
                A[i][j] = numerator/denominator
        
    O = np.zeros(shape=(num_states, num_obs))
    for o in range(num_obs):
        for state in range(num_states):
            denominator = 0
            numerator = 0
            for length in range(len_):
                indicator = 0
                if obs[length] == o:
                    indicator = 1
                numerator += gamma[length][state] * indicator
                denominator += gamma[length][state]
                O[state][o] = numerator/denominator
       
    return A, O

if __name__ == "__main__":
    main()
