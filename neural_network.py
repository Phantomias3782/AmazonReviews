# Provide all functions for the neural network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(train_data, test_data, val_perc = 0.05):
    "prepare data for neural network, reshape it"

    # build train data
    x_train = train_data["word_vectors"].to_numpy()
    y_train = train_data["overall"].to_numpy()

    # build test data
    x_test = test_data["word_vectors"].to_numpy()
    y_test = test_data["overall"].to_numpy()

    # list of tuples to list of lists
    x_train = [list(sub_tuple) for sub_tuple in x_train]
    x_test = [list(sub_tuple) for sub_tuple in x_test]

    # reformat to numpy array
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # shape
    shape = len(x_train[1:2][0])

    # reshape data
    x_train, y_train, x_test, y_test = reshape(x_train, y_train, x_test, y_test, shape)

    # build validation_data
    scope = round(len(x_train)*val_perc)

    x_val = x_train[:scope]
    x_train = x_train[scope:]

    y_val = y_train[:scope]
    y_train = y_train[scope:]

    # return data
    return x_train, y_train, x_val, y_val, x_test, y_test, shape

def reshape(x_train, y_train, x_test, y_test, shape):
    "reshape data"

    x_train = x_train.reshape(len(x_train), shape).astype("float32") / 255
    x_test = x_test.reshape(len(x_test), shape).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    return x_train, y_train, x_test, y_test

def build_model(shape, neuron_layers = 1, neuron_count = [20], activation_functions = ["sigmoid"]):
    "neuron count should contain as list number of neurons per layer"

    # build model
    inputs = keras.Input(shape=(shape,), name="wordvecors")
    first = True
    
    # dynamically add layers
    for i in range(neuron_layers):

        if first:
            
            x = layers.Dense(neuron_count[i], activation= activation_functions[i], name="dense_"+str(i))(inputs)

            # set first to false
            first = False

        else:

            x = layers.Dense(neuron_count[i], activation= activation_functions[i], name="dense_"+str(i))(x)

    # add output layer
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    # return model
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size = 64, show_output = True):
    "train model with given parameters"

    print("Started training.")

    history = model.fit(
        x_train,
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = show_output,
        validation_data = (x_val, y_val)
    )

    print("Finished training.")

    return model, history.history

def evaluate_model(model, x_test, y_test):
    "returns test loss and test accuracy"

    results = model.evaluate(x_test, y_test, batch_size=128)
    
    return results

def plot_training_progress(history):
    "plot training progress of history variable"

    # plot train and val accuracy
    plt.plot(history['sparse_categorical_accuracy'])
    plt.plot(history['val_sparse_categorical_accuracy'])

    # plot settings
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

def model_predict(input_vector, model):
    "predict class"
    
    # get softmax array
    result = model.predict(input_vector)
    
    # get predicted class
    pred_class = np.argmax(result, axis = -1)[0]
    
    return pred_class

def prepare_nn_input(word_vector, shape):
    "returns np array in shape needed for nn model. Input is a word vector"
    
    # reshape
    input_vector = np.array(word_vector)
    input_vector = input_vector.reshape(len(input_vector), shape).astype("float32") / 255
    
    return input_vector