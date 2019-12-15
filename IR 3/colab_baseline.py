import pandas as pd
import numpy as np
import pickle, operator
from copy import deepcopy
from collections import Counter
from time import time
import sklearn.metrics.pairwise as pw

def RMSE(pred,value):
    N = len(pred)
    sum = np.sum(np.square(pred-value))
    return np.sqrt(sum/N)
	
def MAE(pred,value):
    N = len(pred)
    sum = np.sum(np.abs(pred-value))
    return sum/N

def main():
    '''
        Pickle library is used to load train and test matrices, users_map and movies_map saved from preprocess.py
    '''
    N_neighbour = 80
    filehandler = open("rating_matrix", 'rb+')
    matrix = pickle.load(filehandler)
    n_users = matrix.shape[0]
    n_movies = matrix.shape[1]
    # print(n_movies, n_users)

    filehandler = open("test_data_dump", 'rb+')
    test = pickle.load(filehandler)
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
 
    matrix = matrix.T	#The rows are now movies and columns are users
    mu = sum(sum(matrix))/np.count_nonzero(matrix)
    '''
    The pearson co-efficient matrix between movies to find similarities between two movies
    '''
    sim_matrix = np.corrcoef(matrix)

    '''
    Using the test cases to estimate the ratings using collaborative filtering (item-item)
    '''
    actual_ratings = []
    pred_ratings = []

    for j in range(len(test["movieid"])):
        user = test.iloc[j,0]
        movie = test.iloc[j,1]
        rating = test.iloc[j,2]
        # print("movie =, user", movie, user)
        movie = movies_map[movie]
        user = users_map[user]
        actual_ratings.append(int(rating))
        '''
            Caluculating the N_neighbour for each movie
        '''
        sim_movie = {}
        i = 0
        for x in sim_matrix[movie]:
        	sim_movie[x] = i
        	i+=1
        sorted_sim = dict(sorted(sim_movie.items(), key=operator.itemgetter(0),reverse=True))
        sim_movie = []
        i = 0
        for x, y in sorted_sim.items():
            if i <= N_neighbour:
                if matrix[y][user] != 0:
                    sim_movie.append(y)
                    i += 1
            else:
                break
        sim_movie = sim_movie[1:]
        # print(len(sim_movie))
        """
            Caluculating predicted raings using baseline 
        """
        b_movie = movies_mean[movie] - mu
        b_user = users_mean[user] - mu
        b = mu + b_user + b_movie
        
        num = 0
        den = 0
        for x in sim_movie:
            if matrix[x][user] != 0:
                bi = b_user + movies_mean[x]
                num += sim_matrix[movie][x] * (matrix[x][user] - bi)
                den += sim_matrix[movie][x]
        temp_rating = b + num/den
        if temp_rating > 5:
            temp_rating = 5
        elif temp_rating < 0:
            temp_rating = 0

        pred_ratings.append(temp_rating)
        if j % 1000 == 0:
            print(j)
    end_time = time()

    print(RMSE(np.array(pred_ratings), np.array(actual_ratings)))
    print(MAE(np.array(pred_ratings), np.array(actual_ratings)))
    print(end_time - start_time)

if __name__ == '__main__':
	main()