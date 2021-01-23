import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import json
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

# custom files
from word_vectorization import create_tf_vector_all_reviews, create_idf_dictionary, create_word_vector_dict
from data_transformation import get_overview, clean_data, balancing

# read full data
path = "./data/AMAZON_FASHION_FULL.json"

data = pd.read_json(path, lines = True)

keepColumns = ["overall", "reviewText"]
data = clean_data(data, keepColumns)

data = balancing(data, balancing = True, graphical = False)

# Create Word Vectors

# create review_dict
review_dict = create_tf_vector_all_reviews(data)
print("Review dict done")
rev_file = open("review_data.json", "w")
json.dump(review_dict, rev_file)
rev_file.close()

# create idf dict
idf_dict = create_idf_dictionary(review_dict)
print("idf dict done")
idf_file = open("idf_data.json", "w")
json.dump(idf_dict, idf_file)
idf_file.close()

# create word_vec_dict
word_vec_dict = create_word_vector_dict(review_dict, idf_dict)
print("word vec dict done")
word_file = open("word_data.json", "w")
json.dump(word_vec_dict, word_file)
word_file.close()

# include word_vec_dict in dataframe
data["word_vectors"] = word_vec_dict.values()
data.to_json("data_with_Words.json")