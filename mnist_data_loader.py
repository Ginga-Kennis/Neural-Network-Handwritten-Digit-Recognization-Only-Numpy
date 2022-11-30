from keras.datasets import mnist
import numpy as np



def create_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    all_training_data = []
    for input,output in zip(train_X,train_y):
        input_data = np.reshape(input,(784,1))
        input_data = input_data/255
        output_data = vertorized_result(output)
        training_data = (input_data,output_data)

        all_training_data.append(training_data)



    all_test_data = []
    for test,output in zip(test_X,test_y):
        test_input_data = np.reshape(test,(784,1))
        test_input_data = test_input_data/255
        training_data = (test_input_data,output)
        all_test_data.append(training_data)



    return (all_training_data,all_test_data)

def vertorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1
    return e

create_data()







