# vectorize all tetxs
## use tf and idf functions of Document finding (previous Lecture)
import math
import numpy as np
import pandas as pd

def check_word(word):
    "Check if word is valid due to some restrictions"
    
    exception_list = ["\n"]
    
    for element in exception_list:
        
        if element in word:
            
            return False
    
    if len(word) < 4:
        
        return False
    
    if word.isalpha():
    
        return True
    
    else:
        
        return False

def create_tf_vector(data):
    "Create tf_vector as dict"
    
    words = data.split(" ")
    
    tmp_dict = {}
    
    count = 0
        
    for word in words:
        
        word = word.lower()
        
        if check_word(word):
            
            count += 1

            if word in tmp_dict.keys():

                tmp_dict[word] += 1

            else:

                tmp_dict[word] = 1
    
    for key in tmp_dict.keys():
        
        tmp_dict[key] = tmp_dict[key] / count
    
    return tmp_dict

def create_tf_vector_all_reviews(dataframe, column = "reviewText"):
    "create a dictionary for each review with the corresponding tf as value"

    # initialize variables
    index = 0
    tf_dict = {}
    
    # get all reviewTexts
    reviewTexts = list(dataframe[column])
    
    # create tf dicts
    for element in reviewTexts:
        
        tf_dict[index] = create_tf_vector(element)
        index += 1
    
    return tf_dict

def create_idf_dictionary(tf_dict):

    # create idf dictionary

    idf_dict = {}

    document_count = 0

    for review_dict in tf_dict.keys():

        document_count += 1
        
        key_list = list(tf_dict[review_dict].keys())

        for list_word in key_list:

            if list_word in idf_dict:

                idf_dict[list_word] += 1

            else:

                idf_dict[list_word] = 1


    for key in idf_dict.keys():

        idf_dict[key] = math.log(document_count / idf_dict[key])
    
    return idf_dict

def create_word_vector(idf_dict, tf_dict):
    
    tmp_word_vec_dict = {}
    
    for idf in idf_dict.keys():
            
            if idf in tf_dict.keys():
                
                tmp_word_vec_dict[idf] = tf_dict[idf] * idf_dict[idf]
            
            else:
                
                tmp_word_vec_dict[idf] = 0
    
    sorted_word_vec = dict(sorted(tmp_word_vec_dict.items()))
    
    return sorted_word_vec

def create_word_vector_dict(review_dict, idf_dict):

    word_vector_dict_textual = {} # can be depriciated lateron
    word_vector_dict_array = {}

    for review_index in review_dict.keys():

        word_vector_dict_textual[review_index] = create_word_vector(idf_dict, review_dict[review_index]) # can be depriciated lateron
        word_vector_dict_array[review_index] = np.array(
            pd.Series(
                create_word_vector(idf_dict, review_dict[review_index])
            )
        )
    
    return word_vector_dict_array

def create_input_word_vector(input_str, idf_dict):
    "create word vector of input string"

    input_tf = create_tf_vector(input_str)

    # create word vector

    word_vec = create_word_vector(idf_dict, input_tf)

    word_vec = [list(word_vec.values())]
    
    return word_vec