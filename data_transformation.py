# Functions for Data viewing and Data Transofrmation
import matplotlib.pyplot as plt

def get_overview(dataframe):
    "Get dataframe overview"

    # get data
    data_len = len(dataframe)
    column_count = len(dataframe.columns)
    columns = list(dataframe.columns)
    columntypes = {}
    
    for col in columns:

        columntypes[col] = str(dataframe[col].dtype)

    # print data
    print(f"Data has {data_len} datapoints.")
    print(f"Data has {column_count} columns.")

    print("Columns: dtype")
    print(columntypes)

def clean_data(dataframe, keepColumns):
    "Clean Dataframe"

    # get all dropable columns
    columns = list(dataframe.columns)
    drop_columns = [i for i in columns if i not in keepColumns]
    drop_len = len(drop_columns)

    # drop unnedded columns
    dataframe.drop(drop_columns, axis=1, inplace=True)

    # drop null values
    data_len_org = len(dataframe)
    dataframe = dataframe.dropna()
    data_len_aft = len(dataframe)

    nan_count = data_len_org - data_len_aft

    # print info
    print(f"Dropped {drop_len} columns.")
    print(f"Droppend {nan_count} null values.")
    print(f"Data contains now {len(dataframe)} datapoints.")

    return dataframe

def balancing(dataframe, balancing = True, graphical = True, max_num = None):
    "Returns information about data balancing and if wished balancd dataset."

    # get all classes and init dict
    rating_count = {}
    all_ratings = sorted(dataframe["overall"].unique())

    # get numbers of rating
    for rating in all_ratings:
        
        rating_count[rating] = len(dataframe[dataframe["overall"] == rating])

    # data information
    most_rated = max(rating_count, key = rating_count.get)
    least_rated = min(rating_count, key = rating_count.get)
    print(f"Most saved rating is {most_rated} with {rating_count[most_rated]} datapoints.")
    print(f"Least saved rating is {least_rated} with {rating_count[least_rated]} datapoints.")

    if graphical:
        # show graph
        plt.bar(list(rating_count.keys()), list(rating_count.values()))
        plt.show()

    # balance Dataset
    if balancing:

        if max_num:

            data_balanced = dataframe[dataframe["overall"] == least_rated][:max_num]
            remain_ratings = [i for i in all_ratings if i != least_rated]


            for remain_rating in remain_ratings:
                
                # append data
                data_balanced = data_balanced.append(dataframe[dataframe["overall"] == remain_rating][:max_num])

            # reset index
            data_balanced = data_balanced.reset_index(drop = True)

            # print dataset length
            print(f"Dataset now contains {len(data_balanced)} datapoints.")

            # return
            return data_balanced
        
        else:

            data_balanced = dataframe[dataframe["overall"] == least_rated]
            remain_ratings = [i for i in all_ratings if i != least_rated]


            for remain_rating in remain_ratings:
                
                # append data
                data_balanced = data_balanced.append(dataframe[dataframe["overall"] == remain_rating][:rating_count[least_rated]])

            # reset index
            data_balanced = data_balanced.reset_index(drop = True)

            # print dataset length
            print(f"Dataset now contains {len(data_balanced)} datapoints.")

            # return
            return data_balanced