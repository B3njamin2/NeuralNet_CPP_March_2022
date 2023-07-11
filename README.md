# Neural Network from Scratch

This project involves building a fully connected neural network with different activation functions from scratch using various libraries such as vector, file processing, random, string, stream, and cmath. The neural network is designed to handle different sizes of input, output, and hidden layers, and supports class inheritance for using different activation functions like LeakyRelu, Sigmoid, and Tanh.

## Core Competencies

The core competencies of this project include:

- Researching and understanding neural network concepts
- Problem assessment and analysis for designing the neural network
- Implementing various functionalities such as weight initialization, forward and back propagation using gradient descent, and import/export of file information

## Getting Started

To use the neural network, follow these steps:

1. Set up the neural network layout by specifying the number of nodes in each layer. For example, `std::vector<int> map = {2, 8, 4};` represents a neural network with 2 input nodes, 8 nodes in the hidden layer, and 4 output nodes.

2. Choose one activation function from LeakyRelu, Sigmoid, or Tanh. Uncomment the desired line to select the activation function. For example, `LeakyRelu net(map);` creates a neural network with the LeakyRelu activation function.

3. Set the learning rate using `net.setLearningRate(0.2);`. This determines the step size for updating the weights during training.

3.5. **Only for LeakyRelu**: If you are using the LeakyRelu activation function, you can set a constant value using `net.setConstant(0.5);`. This constant affects the negative part of the LeakyRelu function.

4. Initialize the weights of the neural network using `net.weightIntialization();`.

5. Train the network by providing a training file and specifying whether to export weight and node data along with prediction results. For example, `net.readandTrain("trainingAllGates.txt", "_outputDataAllGates.txt", 0);` reads the training data from "trainingAllGates.txt", performs training, and only exports prediction results.

6. Optionally, you can test the neural network by providing inputs through the terminal and verifying the outputs using `net.test();`.

## Available Functions

The following functions are available for interacting with the neural network:

- `void weightIntialization()`: Initializes the weights of the neural network.
- `void forwardProp(const std::vector<double> &inputs)`: Performs forward propagation for the given inputs.
- `void backProp(const std::vector<double> &targets)`: Performs backpropagation using the given target outputs.
- `std::vector<double> getOutput()`: Retrieves the output of the neural network.
- `double costFunction(const std::vector<double> &outputs, const std::vector<double> &targets) const`: Calculates the cost function given the outputs and target outputs.
- `void importWeights(std::string filename)`: Imports the weights from a file.
- `void exportWeights(std::string filename) const`: Exports the weights to a file.
- `void exportNodeInfo(std::string filename)`: Exports the node information to a file.
- `static void setLearningRate(double Rate)`: Sets the learning rate for the neural network.
- `void readandTrain(std::string trainingFile, std::string OutputFileName, int outputData)`: Reads the training data from a file, performs training, and exports results based on the specified options.
- `void test()`: Allows testing the neural network by providing inputs through the terminal and verifying the outputs.

##Testing and Troubleshooting
The program includes multiple pre-existing training data files that can be used for testing. However, you can also test the neural network with your own training data as long as it is in a TXT file. Ensure that the inputs and targets are separated by a space, and each epoch's data is on a new line. The program validates the input and target sizes against the neural network to ensure compatibility.

In case of issues with exploding or vanishing gradients, it is recommended to rerun the program as random weight initialization can sometimes cause such problems.
