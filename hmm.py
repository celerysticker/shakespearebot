import numpy as np

# Reference: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    
def train_hmm(train_data, num_states, num_obs, A, O):
    """Takes data as input and returns a HMM modeled (A, O matrices)
    using unsupervised training"""
    converged = False
    
    iteration = 0
    while iteration is not 1: # run the training once
        print "Iteration " + str(iteration)
        A_old, O_old = A, O
        
        L =  len(train_data)
        alphas = []
        betas = []
        
        for l,line in enumerate(train_data):
            alpha, c = Forward(line, num_states, A, O)
            alphas.append(alpha)
            
            beta = Backward(line, num_states, A, O, c) #c is the normalization constant
            betas.append(beta)
            
            if l == 1:
                print len(alpha), len(alpha[0])
                print "--------------------------"
                print beta
            #gamma, xi = E_step()
            
        #A, O = M_step() # sum over all training examples
        #A, O = EM_step(train_data, num_states, num_obs, A, O)
        
        # check convergence condition
        A_norm = np.linalg.norm(A - A_old, 'fro')
        O_norm = np.linalg.norm(O - O_old, 'fro')
        print "Norm of diff of A: " + str(A_norm)
        print "Norm of diff of O: " + str(O_norm)
        
        iteration += 1
    return A, O

def Forward(obs, num_states, A, O):
    len_ = len(obs)
    # Forward subalgorithm (computed recursively)
    alpha = [[[0.] for i in range(num_states)] for t in range(len_)]
    alpha[0] = [(1. / num_states) * O[i][obs[0]] for i in range(num_states)]
    
    c = [[0.] for t in range(len_)]
    c[0] = 1 / sum(alpha[0])

    alpha[0] = [alpha[0][i] / c[0] for i in range(num_states)] #renormalize alpha
    
    for t in range(1, len_):
        for j in range(num_states):
            p_obs = O[j][obs[t]] # probability of observing data in  our given 'state'
            p_trans = 0  # probability of transitioning to 'state'
            for i in range(num_states): # iterate through all possible prev states
                p_trans += alpha[t - 1][i] * A[i][j]
            alpha[t][j] = p_trans * p_obs # update probability
        c[t] = 1 / sum(alpha[t])
        alpha[t] = [alpha[t][i] * c[t] for i in range(num_states)] #renormalize alpha
    return alpha, c
    
def Backward(obs, num_states, A, O, c):
    len_ = len(obs)
    # Backward subalgorithm 
    beta = [[[0.] for i in range(num_states)] for i in range(len_)] 
    beta[len_ - 1] = [1 * c[len_ -1] for i in range(num_states)]
    
    for t in range(len_ - 2, -1, -1):
        for i in range(num_states):
            p_end = 0
            for j in range(num_states):
                p_end += beta[t + 1][j] * A[i][j] * O[j][obs[t + 1]]
            beta[t][i] = p_end * c[t]
    return beta

def E_step():
    pass
    
def M_step():
    pass
    
    
    
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

