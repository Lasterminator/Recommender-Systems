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

    '''
    The mean rating of each user is calculated
    '''
    users_mean = matrix.sum(axis=1)
    counts = Counter(matrix.nonzero()[0])
    for i in range(n_users):
        if i in counts.keys():
            users_mean[i] = users_mean[i]/counts[i]
        else:
            users_mean[i] = 0

    '''
    The mean rating of each movie is calculated
    '''
    movies_mean = matrix.T.sum(axis=1)
    counts = Counter(matrix.T.nonzero()[0])
    for i in range(n_movies):
        if i in counts.keys():
            movies_mean[i] = movies_mean[i]/counts[i]
        else:
            movies_mean[i] = 0

    '''
    The probabilities of selection along the columns and rows is calculated
    '''
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
    selected_col = np.random.choice(np.arange(0,n_movies), size = c, replace=False, p = col_norm)
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
    selected_row = np.random.choice(np.arange(0,n_users), size = c, replace=False, p = row_norm)
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
    for i in range(W_cur.shape[0]):
        W_cur[i][i] = 1/W_cur[i][i]
    U = np.dot(np.dot(W2.T, W_cur**2), W1.T)
    cur_100 = np.dot(np.dot(C, U), R.T)

    '''
    All ratings estimated to be greater than 5 or less than 0 are rewritten
    '''
    for i in range(cur_100.shape[0]):
        for j in range(cur_100.shape[1]):
            if cur_100[i,j] > 5:
                cur_100[i,j] = 5
            elif cur_100[i,j] < 0:
                cur_100[i,j] = 0
    end_time = time()
    # print(cur_100[:10,:10])
    print(RMSE(cur_100, matrix))
    print(MAE(matrix, cur_100))
    print(end_time - start_time)

if __name__ == '__main__':
    main()