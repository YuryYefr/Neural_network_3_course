# Neural Network Course Project

This repository contains materials and implementations for a Neural Network course. The project demonstrates fundamental concepts in neural networks through practical implementations.

## Lab 1: XOR Problem with Perceptron

The first lab implements a simple Perceptron to solve the XOR problem with 4 variables. The XOR (exclusive OR) problem is a classic example used in neural network education to demonstrate that a single-layer perceptron cannot solve non-linearly separable problems.

In this implementation:
- The perceptron is trained to determine if an input of 4 binary values has an odd or even number of 1's
- Output is 1 if there's an odd number of 1's, and 0 if there's an even number
- The implementation includes a simple Perceptron class with methods for activation, prediction, and training

## Lab 2: Elman Backpropagation Neural Network for Function Approximation

The second lab implements an Elman backpropagation neural network to model a function of two variables. The Elman network is a type of recurrent neural network that has feedback connections from the hidden layer to a context layer, providing the network with memory.

In this implementation:
- The network models the function f(x,y) = x² + y² in the range [0, 10]
- Two network configurations are compared:
  1. 1 hidden layer with 15 neurons
  2. 3 hidden layers with 5 neurons each
- The implementation includes:
  - Data generation and normalization
  - Complete Elman network implementation with forward and backward passes
  - Training with mini-batch gradient descent
  - Evaluation metrics including MSE, MAE, and relative error
  - Visualization of training progress and results

## Lab 3: Feedforward Neural Network for Handwritten Digit Recognition

The third lab implements a feedforward neural network for recognizing handwritten digits using the MNIST dataset. This implementation follows the structure described in the task, where each layer's outputs are fully connected to all inputs of the next layer's elements, and the activation function is the same for all elements in the network.

In this implementation:
- The network is trained to recognize handwritten digits (0-9) from the MNIST dataset
- Two network configurations are compared:
  1. 1 hidden layer with 128 neurons
  2. 2 hidden layers with 64 neurons each
- The implementation includes:
  - Loading and preprocessing the MNIST dataset
  - Complete feedforward neural network implementation with forward and backward passes
  - Training with mini-batch gradient descent and cross-entropy loss
  - Evaluation metrics including accuracy and classification report
  - Visualization of training progress, confusion matrices, and example predictions

## Installation

To run the notebooks in this repository, you need to have Python installed along with the required dependencies.

1. Clone this repository:
```
git clone https://github.com/yourusername/Neural_network_3_course.git
cd Neural_network_3_course
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Dependencies

- NumPy ~= 2.0.1
- Matplotlib >= 3.5.0
- scikit-learn >= 1.6.1

## Usage

To run the Jupyter notebooks:

1. Start Jupyter Lab or Jupyter Notebook:
```
jupyter lab
```
or
```
jupyter notebook
```

2. Open the desired notebook (e.g., `lab_1.ipynb`, `lab_2.ipynb`, or `lab_3.ipynb`) and run the cells.

## Perceptron Implementation

The Perceptron implementation in Lab 1 includes:

- Initialization with random weights and bias
- Configurable learning rate and number of epochs
- Step activation function
- Training method that updates weights and bias based on prediction error
- Testing functionality to evaluate the trained model

## Elman Network Implementation

The Elman backpropagation neural network implementation in Lab 2 includes:

- Recurrent neural network architecture with context layer for memory
- Configurable number of hidden layers and neurons per layer
- Sigmoid activation function with its derivative for backpropagation
- Mini-batch training with configurable batch size and epochs
- Context layer that stores the previous state of the hidden layer
- Comprehensive evaluation metrics (MSE, MAE, relative error)
- Visualization tools for analyzing network performance
- Comparison of different network architectures

## Feedforward Neural Network Implementation

The Feedforward Neural Network implementation in Lab 3 includes:

- Fully connected multilayer neural network architecture
- Configurable number of hidden layers and neurons per layer
- Sigmoid activation function for hidden layers and softmax for output layer
- Mini-batch training with configurable batch size and epochs
- Forward and backward propagation with gradient descent optimization
- Cross-entropy loss function for classification
- Comprehensive evaluation metrics (accuracy, classification report, confusion matrix)
- Visualization tools for analyzing network performance and predictions
- Comparison of different network architectures on the MNIST dataset

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines if applicable]
