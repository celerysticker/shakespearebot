import numpy as np

# Reference: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    
def train_hmm(train_data, num_states, num_obs, A, O):
    """Takes data as input and returns a HMM modeled (A, O matrices)
    using unsupervised training"""
    converged = False
    
    iteration = 0
    while iteration is not 1: # fixed number of iterations to be 10
        print "Iteration " + str(iteration)
        A_old, O_old = A, O
        
        A, O = EM_step(train_data, num_states, num_obs, A, O)
        
        # check convergence condition
        A_norm = np.linalg.norm(A - A_old, 'fro')
        O_norm = np.linalg.norm(O - O_old, 'fro')
        print A_norm
        print O_norm
        iteration += 1
    return A, O

def forward():
    return alpha
    
def bac
    
def EM_step(train_data, num_states, num_obs, A, O):
    """Runs Forward-Backward algorithm to compute marginal probabilities """
        
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

