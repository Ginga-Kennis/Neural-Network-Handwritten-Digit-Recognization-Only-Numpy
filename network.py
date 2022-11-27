import random
import numpy as np

class Network(object):

    def __init__(self,sizes):
        self.num_layers = len(sizes)  #3
        self.sizes = sizes  #[784 30 10]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]  #[ [30*1] [10*1] ]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]  #[ [30*784] [10*30] ]


    def feedforward(self,a):
        #a = input layer [784*1]
        #calculates the output layer activation
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
        #a = output layer [10*1]

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        #Stochastic Gradient Discent(確率的勾配降下法)
        #epochs = How many times to use the whole training_data,何回トレーニングデータを使い切ったか
        #eta = training rate
        if test_data:
            n_test = len(test_data)  #number of test data
        n = len(training_data)  #number of training data
        for j in range(epochs):
            random.shuffle(training_data)
            #create mini batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            #for each mini_batch apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)  #update the network weights and biases

            #print(self.biases)
            #print(self.weights)
            #Printing the training rate after each epoch
            if test_data:
                print(f'Epoch {j} : {self.evaluate(test_data)} / {n_test} ')
            else:
                print(f"Epoch {j} complete")


    #updates the network weights and biases according to a single iteration of gradient descent, using just the training data in mini_batch.
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  #∇b
        nabla_w = [np.zeros(w.shape) for w in self.weights]  #∇w
        for x, y in mini_batch:
            #calculating the gradients
            delta_nabla_b,delta_nabla_w = self.backprop(x,y) #dC/db,dC/dw
            #update gradients
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b,nb in zip(self.biases, nabla_b)]


    def backprop(self,x,y):
        #x→training input[784*1],y→desired output[10*1]
        #returnung a tuple (dC/db,dC/dw) representing the gradient for the cost function
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  #list to store all the activations,layer by layer
        zs = []  #list to store all the z vactors, layer by layer

        #feedforward
        for b,w in zip(self.biases,self.weights):
            #calculate the next layer
            z = np.dot(w,activation)+b
            zs.append(z)
            #update activation to the next layer
            activation = sigmoid(z)
            activations.append(activation)


        #backward pass
        #calculate the cost for the output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #backpropagate error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        #print(nabla_b)
        #print(nabla_w)
        return (nabla_b, nabla_w)

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self,output_activations,y):
        return (output_activations - y)




def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))




