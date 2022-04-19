import os

#------ Make sure parallel runs don't explode CPU --------#
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

#------ Import Libraries --------#
import NNTucktools  #this .py file contains function definitions for non_negative_tucker with KL loss,
                    # as well as for MULTITENSOR's EM updates
import numpy as np
from NNTucktools import non_negative_tucker, non_negative_tucker_ones
import tensorly as tl
from sklearn.metrics import roc_auc_score
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.base import unfold
import numpy as np
from joblib import parallel_backend
from joblib import Parallel, delayed
NUM_IT = 50
def masking_tensor(tensor):
    # This function returns 5 masking tensors for a **directed** network (e.g., M != M.T)
        # here, M_ijk == 0 means that A_ijk is _unobserved_ 
        # In each of the M's with uniform probability a 
        # random set of ~20% of the entries are unobserved
        
    ALPHA, N, N = np.shape(tensor)
    M_fold = np.random.rand(ALPHA, N, N)
    
    M_1 = np.ones_like(tensor)
    M_2 = np.ones_like(tensor)
    M_3 = np.ones_like(tensor)
    M_4 = np.ones_like(tensor)
    M_5 = np.ones_like(tensor)

    M_1[np.where(M_fold<=0.2)] = 0
    M_2[np.where(np.logical_and(M_fold<=0.4, M_fold>0.2))] = 0
    M_3[np.where(np.logical_and(M_fold<=0.6, M_fold>0.4))] = 0
    M_4[np.where(np.logical_and(M_fold<=0.8, M_fold>0.6))] = 0
    M_5[np.where(M_fold > 0.8)] = 0

    Maskings = [M_1, M_2, M_3, M_4, M_5]
    return Maskings

def masking_tensor_undirected(tensor): 
    # This function returns a masking tensor for a **directed** network (e.g., M == M.T)
        # here, M_ijk == 0 means that A_ijk is _unobserved_ 
        # In each of the M's with uniform probability a 
        # random set of ~20% of the entries are unobserved
        # here if M_ijk is unobserved, so is M_jik
        
    M_1 = np.ones_like(tensor)
    M_2 = np.ones_like(tensor)
    M_3 = np.ones_like(tensor)
    M_4 = np.ones_like(tensor)
    M_5 = np.ones_like(tensor)
    
    ALPHA, N, N = np.shape(tensor)
    M_fold = np.random.rand(ALPHA, N, N)

    M_1[np.where(np.logical_and(np.triu(M_fold) <= 0.2, np.triu(M_fold) > 0.0))] = 0
    a=np.where(M_1 == 0)
    b = tuple((a[0], a[2], a[1]))
    M_1[b] = 0

    M_2[np.where(np.logical_and(np.triu(M_fold) <= 0.4, np.triu(M_fold) > 0.2))] = 0
    a=np.where(M_2 == 0)
    b = tuple((a[0], a[2], a[1]))
    M_2[b] = 0

    M_3[np.where(np.logical_and(np.triu(M_fold) <= 0.6, np.triu(M_fold) > 0.4))] = 0
    a=np.where(M_3 == 0)
    b = tuple((a[0], a[2], a[1]))
    M_3[b] = 0

    M_4[np.where(np.logical_and(np.triu(M_fold) <= 0.8, np.triu(M_fold) > 0.6))] = 0
    a=np.where(M_4 == 0)
    b = tuple((a[0], a[2], a[1]))
    M_4[b] = 0

    M_5[np.where(np.logical_and(np.triu(M_fold) <= 1, np.triu(M_fold) > 0.8))] = 0
    a=np.where(M_5 == 0)
    b = tuple((a[0], a[2], a[1]))
    M_5[b] = 0

    Maskings = [M_1, M_2, M_3, M_4, M_5]
    return Maskings

def Per_alpha(tensor, c, K):
    # for Y of dimension (L x c) sweep over dimension c
        # for each c, define a set of 5 masking tensors
        # then for each M_i in this set:
            # return the decompositon associated with the highest likelihood over num_it runs
        # returns max_results which is a list of length 5 where each entry is an
        # object of size (4):(current_AUC, current_like, ten, M)
        # each item in max_results is the info for the NTD with the highest log like/AUC for each M
        
    print("doing c = {}".format(c))
    Maskings = masking_tensor_undirected(tensor) #malaria is undirected. 
    num_it = NUM_IT
    with parallel_backend('multiprocessing'):
        Big_Results = Parallel(n_jobs = 37)(delayed(Per_j)(j, Maskings, tensor, c, K, symm = True) for j in range(num_it))
    #Big_Results is (num_it x 5) where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    likelihoods = np.zeros((num_it, 5))
    AUCs = np.zeros((num_it, 5))
    for i in range(num_it):
        for j in range(5):
            likelihoods[i, j] = Big_Results[i][j][1]
            AUCs[i, j] = Big_Results[i][j][0]
    max_results = []
    goal = likelihoods # change to AUCs if you want to pick the decomp with the highest test AUC
    for j in range(5):
        max_idx = np.argmax(goal[:, j])
        max_results.append(Big_Results[max_idx][j])
    return max_results 

def Per_alpha_c(tensor, K):
    # for Y == I
        # define a set of 5 masking tensors
        # then for each M_i in this set:
            # return the decompositon associated with the highest likelihood over num_it runs
        # returns max_results which is a list of length 5 where each entry is an
        # object of size (4):(current_AUC, current_like, ten, M)
        # each item in max_results is the info for the NTD with the highest log like/AUC for each M
        
    ALPHA, N, N = np.shape(tensor)
    c = ALPHA
    num_it = NUM_IT
    print("doing Y == I")
    Maskings = masking_tensor_undirected(tensor)
    with parallel_backend('multiprocessing'):
        Big_Results = Parallel(n_jobs = 37)(delayed(Per_j_c)(j, Maskings, tensor, c, K, symm = True) for j in range(num_it))
    #Big_Results is (num_it x 5) where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    
    likelihoods = np.zeros((num_it, 5))
    AUCs = np.zeros((num_it, 5))
    for i in range(num_it):
        for j in range(5):
            likelihoods[i, j] = Big_Results[i][j][1]
            AUCs[i, j] = Big_Results[i][j][0]
    max_results = []
    goal = likelihoods # change to AUCs if you want to pick the decomp with the highest test AUC
    for j in range(5):
        max_idx = np.argmax(goal[:, j])
        max_results.append(Big_Results[max_idx][j])
    return max_results 

def Per_alpha_ones(tensor, K):
    # for Y == I
        # define a set of 5 masking tensors
        # then for each M_i in this set:
            # return the decompositon associated with the highest likelihood over num_it runs
        # returns max_results which is a list of length 5 where each entry is an
        # object of size (4):(current_AUC, current_like, ten, M)
        # each item in max_results is the info for the NTD with the highest log like/AUC for each M
    print('K is ', K)
    ALPHA, N, N = np.shape(tensor)
    c = 1
    num_it = NUM_IT
    Maskings = masking_tensor_undirected(tensor)
    with parallel_backend('multiprocessing'):
        Big_Results = Parallel(n_jobs = 37)(delayed(Per_j_ones)(j, Maskings, tensor, c, K, symm = True) for j in range(num_it))
    #Big_Results is (num_it x 5) where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    
    likelihoods = np.zeros((num_it, 5))
    AUCs = np.zeros((num_it, 5))
    for i in range(num_it):
        for j in range(5):
            likelihoods[i, j] = Big_Results[i][j][1]
            AUCs[i, j] = Big_Results[i][j][0]
    max_results = []
    goal = likelihoods # change to AUCs if you want to pick the decomp with the highest test AUC
    for j in range(5):
        max_idx = np.argmax(goal[:, j])
        max_results.append(Big_Results[max_idx][j])
    return max_results

def Per_k(tensor, K):
    # for Y == I
        # define a set of 5 masking tensors
        # then for each M_i in this set:
            # return the decompositon associated with the highest likelihood over num_it runs
        # returns max_results which is a list of length 5 where each entry is an
        # object of size (4):(current_AUC, current_like, ten, M)
        # each item in max_results is the info for the NTD with the highest log like/AUC for each M
        
    ALPHA, N, N = np.shape(tensor)
    c = ALPHA
    num_it = NUM_IT
    print("doing K = {}".format(K))
    Maskings = masking_tensor_undirected(tensor)
    with parallel_backend('multiprocessing'):
        Big_Results = Parallel(n_jobs = 37)(delayed(Per_j_c)(j, Maskings, tensor, c, K, symm = True) for j in range(num_it))
    #Big_Results is (num_it x 5) where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    
    likelihoods = np.zeros((num_it, 5))
    AUCs = np.zeros((num_it, 5))
    for i in range(num_it):
        for j in range(5):
            likelihoods[i, j] = Big_Results[i][j][1]
            AUCs[i, j] = Big_Results[i][j][0]
    max_results = []
    goal = likelihoods # change to AUCs if you want to pick the decomp with the highest test AUC
    for j in range(5):
        max_idx = np.argmax(goal[:, j])
        max_results.append(Big_Results[max_idx][j])
    return max_results
        
def Per_j(j, Maskings, tensor, C, K, symm = True):
    # for each j in range(num_it) need to do a NTD
        # returns C_Results which is a list of length 5
        # where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    if j%5==0:
        print('j = ', j)
    C_Results = Parallel(n_jobs=1)(delayed(TuckerAUC_masked)(M, tensor, C, K, symm) for M in Maskings)
    return C_Results

def Per_j_c(j, Maskings, tensor, C, K, symm = True):
    # For constrained case
    # for each j in range(num_it) need to do a NTD
        # returns C_Results which is a list of length 5
        # where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    ALPHA, N, N = np.shape(tensor)
    if j%5==0:
        print('j = ', j)
    C_Results = Parallel(n_jobs=1)(delayed(TuckerAUC_masked_c)(M, tensor, ALPHA, K, symm) for M in Maskings)
    return C_Results

def Per_j_ones(j, Maskings, tensor, C, K, symm = True):
    # For constrained case
    # for each j in range(num_it) need to do a NTD
        # returns C_Results which is a list of length 5
        # where each entry is an object of size (4):(current_AUC, current_like, ten, M)
    ALPHA, N, N = np.shape(tensor)
    if j%5==0:
        print('j = ', j)
    C_Results = Parallel(n_jobs=1)(delayed(TuckerAUC_masked_ones)(M, tensor, 1, K, symm) for M in Maskings)
    return C_Results

def TuckerAUC_masked_c(M, tensor, C, K, symm = True):
    # Does a non-negative tucker decomposition with given masking tensor and dimension C
    # For the constrained case Y == I
        # Returns an object of size (4):(current_AUC, current_like, ten, M)
        # Where current_AUC is the _test_ AUC, 
        # current_like is the likelihood defined over the OBSERVED  entries
        # ten is the (core, factors) pair of the NTD
        # M is the masking tensor associated w this NTD
        
        ten, log_like, kl_loss = non_negative_tucker(tensor, rank=[C, K, K],init='random', n_iter_max=1000,
                                                         symmetric = symm, masked = True, Masking = M, 
                                                  MT_stopping_conditions = True, returnErrors = True,
                                                    verbose=False, constrained=True, loss = 'KL', tol=10e-6)
        current_like = log_like[-1];
        core, factors = ten
        m_score = tucker_to_tensor((core, factors), transpose_factors= False)
        m_score = m_score[np.where(M ==0)] #only the unobserved entries
        m_true = tensor[np.where(M==0)] #only the unobserved entries
        current_AUC = roc_auc_score(m_true, m_score)
        return (current_AUC, current_like, ten, M) #we want to keep the core and factors to inspect later!

def TuckerAUC_masked(M, tensor, C, K, symm = True):
    # Does a non-negative tucker decomposition with given masking tensor and dimension C
        # Returns an object of size (4):(current_AUC, current_like, ten, M)
        # Where current_AUC is the _test_ AUC, 
        # current_like is the likelihood defined over the OBSERVED  entries
        # ten is the (core, factors) pair of the NTD
        # M is the masking tensor associated w this NTD
        
        ten, log_like, kl_loss = non_negative_tucker(tensor, rank=[C, K, K],init='random', n_iter_max=1000,
                                                         symmetric = symm, masked = True, Masking = M, 
                                                  MT_stopping_conditions = True, returnErrors = True,
                                                    verbose=False, constrained=False, loss = 'KL', tol=10e-6)
        current_like = log_like[-1];
        core, factors = ten
        m_score = tucker_to_tensor((core, factors), transpose_factors= False)
        m_score = m_score[np.where(M ==0)] #only the unobserved entries
        m_true = tensor[np.where(M==0)] #only the unobserved entries
        current_AUC = roc_auc_score(m_true, m_score)
        return (current_AUC, current_like, ten, M) #we want to keep the core and factors to inspect later!

def TuckerAUC_masked_ones(M, tensor, C, K, symm = True):
    # Does a non-negative tucker decomposition with given masking tensor and dimension C
    # For the constrained case Y == ones(1,L)
        # Returns an object of size (4):(current_AUC, current_like, ten, M)
        # Where current_AUC is the _test_ AUC, 
        # current_like is the likelihood defined over the OBSERVED  entries
        # ten is the (core, factors) pair of the NTD
        # M is the masking tensor associated w this NTD
        
        ten, log_like, kl_loss = non_negative_tucker_ones(tensor, rank=[1, K, K],init='random', n_iter_max = 1000,
                                                         symmetric = symm, masked = True, Masking = M,
                                                  MT_stopping_conditions = True, returnErrors = True,
                                                    verbose=False, constrained=True, loss = 'KL', tol=10e-6)
        current_like = log_like[-1];
        core, factors = ten
        m_score = tucker_to_tensor((core, factors), transpose_factors= False)
        m_score = m_score[np.where(M ==0)] #only the unobserved entries
        m_true = tensor[np.where(M==0)] #only the unobserved entries
        current_AUC = roc_auc_score(m_true, m_score)
        return (current_AUC, current_like, ten, M) #we want to keep the core and factors to inspect la$

