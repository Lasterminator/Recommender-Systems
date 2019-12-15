import numpy as np
import pickle
from collections import Counter
from numpy.linalg import svd
import numpy.linalg as linalg
from time import time
import random

def RMSE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    cur_sum = np.sum(np.square(pred-value))
    return np.sqrt(cur_sum/(N*M))
    
def MAE(pred, value):
    N = pred.shape[0]
    M = pred.shape[1]
    sum1 = np.sum(np.abs(pred-value))
    return (sum1/(N*M))

def main():
    '''
        Pickle library is used to load matrix, users_map and movies_map saved from preprocess.py
    '''
    filehandler = open("rating_matrix", 'rb+')
    matrix = pickle.load(filehandler)
    n_users = matrix.shape[0]
    n_movies = matrix.shape[1]

    filehandler = open("user_dump", 'rb+')
    users_map = pickle.load(filehandler)

    filehandler = open("movie_dump", 'rb+')
    movies_map = pickle.load(filehandler)

    start_time = time()

    total_norm =  np.linalg.norm(matrix)
    col_norm =  np.linalg.norm(matrix,axis = 0)
    row_norm =  np.linalg.norm(matrix,axis = 1)
    for i in range(n_movies):
        col_norm[i] = (col_norm[i]/total_norm)**2
        
    for i in range(n_users):
        row_norm[i] = (row_norm[i]/total_norm)**2
    # print(row_norm, col_norm)
    '''
    Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
    '''
    c=1700
    selected_col = []
    C = np.zeros([n_users,c])
    selected_col = np.random.choice(np.arange(0,n_movies), size = c,replace=False, p = col_norm)
    i=0
    for x in selected_col:
        p = col_norm[x]
        d = np.sqrt(c*p)
        C[:,i] = matrix[:,x]/d
        i = i+1
    '''
    Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
    '''
    r=1700
    selected_row = []
    R = np.zeros([n_movies,r])
    selected_row = np.random.choice(np.arange(0,n_users), size = c,replace=False, p = row_norm)
    i=0
    for x in selected_row:
        p = row_norm[x]
        d = np.sqrt(r*p)
        R[:,i] = matrix.T[:,x]/d
        i = i+1
    '''
    The matrix U is constructed from W by the Moore-Penrose pseudoinverse
    This step involves using SVD to find U and V' of W.
    W is calculated as the intersection of the selected rows and columns
    '''
    W = C[selected_row,:]
    W1, W_cur, W2 = svd(W)
    W_cur = np.diag(W_cur)
   
    total_sum = 0
    dimensions = W_cur.shape[0]
    for i in range(dimensions):
        total_sum = total_sum + np.square(W_cur[i,i])   #Find square of sum of all diagonals
    retained = total_sum
    while dimensions > 0:
        retained = retained - np.square(W_cur[dimensions-1,dimensions-1])
        if retained/total_sum < 0.9:        #90% energy retention
            break
        else:
            W1 = W1[:,:-1:]
            W2 = W2[:-1,:]
            W_cur = W_cur[:,:-1]
            W_cur = W_cur[:-1,:]
            dimensions = dimensions - 1     #Dimensionality reduction

    for i in range(W_cur.shape[0]):
        if W_cur[i][i] != 0:
            W_cur[i][i] = 1/W_cur[i][i]
    U = np.dot(np.dot(W2.T, W_cur**2), W1.T)
    cur_90 = np.dot(np.dot(C, U), R.T)      #A = C*U*R

    '''
    All ratings estimated to be greater than 5 or less than 0 are rewritten
    '''
    for i in range(cur_90.shape[0]):
        for j in range(cur_90.shape[1]):
            if cur_90[i,j] > 5:
                cur_90[i,j] = 5
            elif cur_90[i,j] < 0:
                cur_90[i,j] = 0
    end_time = time()
    print(RMSE(cur_90, matrix))
    print(MAE(matrix, cur_90))
    print(end_time - start_time)

if __name__ == '__main__':
    main()