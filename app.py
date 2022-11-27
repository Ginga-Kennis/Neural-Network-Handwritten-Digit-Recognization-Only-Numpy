from network import Network
import mnist_data_loader



net = Network([784,100,50,10])
training_data,test_data = mnist_data_loader.create_data()
net.SGD(training_data,100,10,0.25,test_data=test_data)

""""""""""
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))
"""""""""





