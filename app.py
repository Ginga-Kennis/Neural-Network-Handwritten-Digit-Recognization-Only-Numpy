from network import Network
import mnist_data_loader

if __name__ == "__main__":
    net = Network([784,100,50,10])
    training_data,test_data = mnist_data_loader.create_data()
    net.SGD(training_data,200,10,0.25,test_data=test_data)








