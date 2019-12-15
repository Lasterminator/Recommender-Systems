import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

'''
Loading the data from a .dat file into a Pandas dataframe
'''
filename = 'data/ratings.dat'
data = []
column_heading = ['userid','movieid','rating','timestamp']
for line in open(filename, 'r'):
    line = line.strip('\n')
    arr = line.split("::")
    arr = [int(x) for x in arr]
    data.append(arr)

rating_mat = pd.DataFrame(data,columns=column_heading)
rating_mat.drop('timestamp',axis=1,inplace=True)
train_data, test_data = train_test_split(rating_mat,test_size=0.1)	#spliting the data into train and test. Testing dats is 10% of total data
'''
Not all movies have been rated and to eliminate discrepancies, the movies and users are mapped
'''
movies_list = rating_mat['movieid'].unique()
no_of_movies = len(movies_list)
movies_map = {}
i = 0
for j in movies_list:
    movies_map[j] = i
    i += 1

users_list = rating_mat['userid'].unique()
no_of_users = len(users_list)
users_map = {}
i = 0
for j in users_list:
    users_map[j] = i
    i += 1

'''
The 2d array, matrix, is the sparse representation of the data, with users as rows and movies as columns
'''
rating_matrix = np.zeros([no_of_users, no_of_movies])
for index, row in rating_mat.iterrows():
    rating_matrix[users_map[row['userid']], movies_map[row['movieid']]] = row['rating']
'''
	Using pickle library to dump python objects onto a file
'''
filehandler = open("rating_matrix",'wb+')
pickle.dump(rating_matrix,filehandler)

filehandler = open("train_data_dump",'wb+')
pickle.dump(train_data,filehandler)

filehandler = open("test_data_dump",'wb+')
pickle.dump(test_data,filehandler)

filehandler = open("user_dump",'wb+')
pickle.dump(users_map,filehandler)

filehandler = open("movie_dump",'wb+')
pickle.dump(movies_map,filehandler)

