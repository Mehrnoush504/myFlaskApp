# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:47:17 2021

@author: Mehrnoush
"""
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import pickle

ratings_dict = {
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1, 3],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3, 0],
}

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))

# Loads Pandas dataframe
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

# save the model to disk
filename = 'svd_model.pkl'
pickle.dump(algo, open(filename, 'wb'))

# load the model from disk
loaded_svd_algo = pickle.load(open('svd_model.pkl', 'rb'))
print(loaded_svd_algo)