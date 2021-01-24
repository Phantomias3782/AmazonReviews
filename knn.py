def knn(train_data, data_point, n):
    "custom implementation of knn"
    
    vectors = train_data["word_vectors"].to_list()
    
    distances = {}
    count = 0
    
    # calculate distance
    for vector in vectors:
        
        distances[count] = distance.euclidean(data_point, np.asarray(vector))
        count += 1
    
    # sort dictionary
    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    
    # use with multiple n
    if isinstance(n, list):

        multiple_n_results = {}

        for sub_n in n:

            multiple_n_results[sub_n] = class_estimation(distances, train_data, sub_n)
        
        # return dict of classes by n's
        return multiple_n_results
    
    elif isinstance(n, int):

        # return class
        return class_estimation(distances, train_data, n)
    


def class_estimation(distances, train_data, n):
    "calculate class"

    classes_indexes = list(distances.keys())[:n]
    classes = {}
    
    for i in classes_indexes:
        
        if train_data.iloc[i]["overall"] in list(classes.keys()):
            
            classes[train_data.iloc[i]["overall"]] += 1
        
        else:
            
            classes[train_data.iloc[i]["overall"]] = 1
    
    max_class = max(classes, key = classes.get)
    
    return max_class

def test(train_data, n_test):

    pred_class = []
    for t_vector in test_vectors:

        pred_class.append(knn(train_data, np.asarray(t_vector), n_test))

    accuracy = accuracy_score(test_classes, pred_class)
    #print(f"accuracy is: {accuracy}")
    return accuracy