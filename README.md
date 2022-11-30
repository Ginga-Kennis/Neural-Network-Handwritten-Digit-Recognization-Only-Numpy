# Neural-Network-Handwritten-Digit-Recognization-Only-Numpy
### I have created a Neural Network to recognize handwritten digits from Scratch,only using Numpy.

## Model for this Neural Network
<p align="center">
  <img src="pics/model.jpg" width="500" />
</p>

## Result
Layers: 784 30 10<br />
Learning Rate: 3<br />
Epochs: 10<br />
### 95% Accuracy

<p align="center">
  <img src="pics/result.png" width="350" />
</p>


# Mathematical Parts 
## Feedforward
This calculates the activation for the next layer.<br />
<p align="center">
  <img src="pics/feedforward.jpg" width="400" />
</p>

## Sigmoid Function
Activation Function I used this time! <br />
<p align="center">
  <img src="pics/sigmoid.jpg" width="400" />
</p>

## Stochastic Gradient Discent Algorithm
1. Loop For Epochs times
   - shuffle training data
   - divide training data into mini batches
   - Loop for each mini batch
     - BackPropagation
   - Update Weights and Biases of the Network using Gradient Descent

## Backpropagation Algorithm
1. Feedforward
2. Output Error: Equation 1
3. BackPropagate Error: Equation 2
4. Calculate δb,δw: Equasion 3,4
<p align="center">
  <img src="pics/4eq.jpg" width="400" />
</p>

