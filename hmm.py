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
        
        alphas, betas, c_array = E_step(train_data, num_states, A, O)
        A, O = M_step(train_data, num_states, num_obs, A, O, alphas, betas, c_array)
    
        #A, O = M_step() # sum over all training examples
        #A, O = EM_step(train_data, num_states, num_obs, A, O)
        
        # check convergence condition
        A_norm = np.linalg.norm(A - A_old, 'fro')
        O_norm = np.linalg.norm(O - O_old, 'fro')
        print "Norm of diff of A: " + str(A_norm)
        print "Norm of diff of O: " + str(O_norm)
        
        iteration += 1
    return A, O
    
def E_step(train_data, num_states, A, O):
    print "Doing E-step"
    alphas = []
    betas = []
    c_array = []
    
    for line in train_data:
        alpha, c = Forward(line, num_states, A, O)
        alphas.append(alpha)
        c_array.append(c)
        
        beta = Backward(line, num_states, A, O, c) #c is the normalization constant
        betas.append(beta)
    return alphas, betas, c_array
   
def M_step(train_data, num_states, num_obs, A, O, alphas, betas, c_array):
    print "Doing M-step"

    A_new = np.random.uniform(size=(num_states, num_states))
    O_new = np.random.uniform(size=(num_states, num_obs))
    
    # update the transition matrix
    for i in range(num_states):
        for j in range(num_states):
            num_sum = 0
            den_sum = 0
            for l, obs in enumerate(train_data):
                len_ = len(obs)
                xi_sum = 0
                gamma_sum = 0
                for t in range(len_ - 1):
                    xi_sum += alphas[l][t][i] * A[i][j] * O[j][obs[t + 1]] * betas[l][t+1][j]
                    gamma_sum += alphas[l][t][i] * betas[l][t][i] / c_array[l][t]
                num_sum += xi_sum 
                den_sum += gamma_sum
            A_new[i][j] = num_sum / den_sum
                    
    
    # update the emission matrix
    for j in range(num_states):
        for k in range(num_obs):
            num_sum = 0
            den_sum = 0
            for l, obs in enumerate(train_data):
                len_ = len(obs)
                indicator_sum = 0
                gamma_sum = 0
                for t in range(len_):
                    if obs[t] == O[j][k]:
                        indicator_sum += alphas[l][t][j] * betas[l][t][j] / c_array[l][t]
                    gamma_sum += alphas[l][t][j] * betas[l][t][j] / c_array[l][t]
                num_sum +=  indicator_sum    
                den_sum += gamma_sum
        O_new[j][k] = num_sum / den_sum
    
    return A_new, O_new
   
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
    beta = [[[0.] for i in range(num_states)] for t in range(len_)] 
    beta[len_ - 1] = [1 * c[len_ -1] for i in range(num_states)]
    
    for t in range(len_ - 2, -1, -1):
        for i in range(num_states):
            p_end = 0
            for j in range(num_states):
                p_end += beta[t + 1][j] * A[i][j] * O[j][obs[t + 1]]
            beta[t][i] = p_end * c[t]
    return beta
