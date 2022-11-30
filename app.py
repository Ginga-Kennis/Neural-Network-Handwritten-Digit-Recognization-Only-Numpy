from network import Network
import mnist_data_loader


net = Network([784,30,10])
training_data,test_data = mnist_data_loader.create_data()
net.SGD(training_data,10,10,3,test_data=test_data)