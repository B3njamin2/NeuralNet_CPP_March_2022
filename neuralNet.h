#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>


std::mt19937 gen(time(0));
std::normal_distribution<double> dis(0, 1);
std::vector<std::vector<double>> readMNISTImages(const std::string &);
std::vector<int> readMNISTLabels(const std::string &);
void updateGraph(const std::vector<double>& costValues, const std::vector<int>& epochs);


// Each the class node owns the weights connected directly in front
class Node
{
    double output;
    double preActOutput;
    double derived;
    std::vector<double> weights;
    Node() {}
    friend class NeuralNet;
};

class NeuralNet
{
protected:
    std::vector<std::vector<Node>> layerArr;
    std::vector<Node> biasArr;
    double learningRate{0.1};

    NeuralNet(const std::vector<int> &map)
    {
        for (size_t i{0}; i < map.size(); i++)
        {
            std::vector<Node> layer;
            for (size_t j{0}; j < map[i]; j++)
                layer.push_back(Node());
            layerArr.push_back(layer);
        }
        for (size_t i{0}; i < map.size() - 1; i++)
        {
            biasArr.push_back(Node());
        }
    }

public:
    void weightIntialization()
    {
        for (size_t i{0}; i < layerArr.size() - 1; i++)
        {
            for (size_t j{0}; j < layerArr[i].size(); j++)
                for (size_t k{0}; k < layerArr[i + 1].size(); k++)
                    layerArr[i][j].weights.push_back(randWeight(layerArr[i].size())); // xavier weight initialization
            biasArr[i].output = 0;
        }
    }

    void forwardProp(const std::vector<double> &inputs)
    {

        if (inputs.size() != layerArr[0].size())
            throw std::runtime_error("input array must be the same size as input nodes!");

        for (size_t i{0}; i < layerArr[0].size(); i++) // load inputs
            layerArr[0][i].output = inputs[i];

        // preactivation
        for (size_t i{1}; i < layerArr.size(); i++)
        {
            for (size_t j{0}; j < layerArr[i].size(); j++)
            {
                double sum = biasArr[i - 1].output; // start the sum with bias
                for (size_t k{0}; k < layerArr[i - 1].size(); k++)
                    sum += (layerArr[i - 1][k].output) * (layerArr[i - 1][k].weights[j]); // sum of input*weights

                layerArr[i][j].preActOutput = sum;

                // activation function
                layerArr[i][j].output = activation(sum);
            }
        }
    }

    void backProp(const std::vector<double> &targets)
    {
        int numOutputs = targets.size();
        if (numOutputs != layerArr.back().size())
            throw std::runtime_error("Target array must be the same size as output nodes!");

        // calculate the derived functions for the output layer
        for (size_t i{0}; i < numOutputs; i++)
        {
            Node *current = &layerArr.back()[i];
            current->derived = derivedActivation(current->preActOutput) * (1.0 / numOutputs) * 2 * (current->output - targets[i]);
        }

        for (int i = layerArr.size() - 2; i >= 0; i--)
        { // all nodes starting from second to last layer
            for (int j{0}; j < layerArr[i].size(); j++)
            {
                Node *current = &layerArr[i][j];

                // calculate the chain rule up to the current node including the activation function and store it
                double derivedSum{0};
                if (i != 0)
                {
                    for (size_t k{0}; k < layerArr[i + 1].size(); k++)
                        derivedSum += current->weights[k] * layerArr[i + 1][k].derived;
                    current->derived = derivedSum * derivedActivation(current->preActOutput);
                }

                // update weights
                for (size_t k{0}; k < layerArr[i + 1].size(); k++)
                    current->weights[k] -= learningRate * current->output * layerArr[i + 1][k].derived;
            }

            // update bias
            double derivedSum{0};
            for (size_t k{0}; k < layerArr[i + 1].size(); k++)
                derivedSum += layerArr[i + 1][k].derived;
            biasArr[i].output -= learningRate * derivedSum;
        }
    }

    std::vector<double> getOutput() const
    {
        std::vector<double> outputs;
        for (size_t i = 0; i < layerArr.back().size(); i++)
            outputs.push_back(layerArr.back()[i].output);

        return outputs;
    }

    double costFunction(const std::vector<double> &outputs, const std::vector<double> &targets) const
    {
        int arrSize = outputs.size();
        if (arrSize != targets.size())
            throw std::runtime_error("Targets and inputs array are not the same size!!");
        double cost{0};
        for (size_t i{0}; i < arrSize; i++)
        {
            cost += pow(targets[i] - outputs[i], 2);
        }
        return cost / arrSize;
    }

    void importWeights(std::string filename)
    {
        std::ifstream inFile(filename, std::ios::in);
        double weight;
        std::string remove;
        for (size_t i{0}; i < layerArr.size() - 1; i++)
        {
            for (size_t j{0}; j < layerArr[i].size(); j++)
                for (size_t k{0}; k < layerArr[i + 1].size(); k++)
                {
                    inFile >> remove >> weight;
                    layerArr[i][j].weights.push_back(weight);
                }
            if (!(inFile >> remove >> weight))
                throw std::runtime_error("Import weights and bias dont match layout of current network");
            biasArr[i].output = weight;
        }
        inFile.close();
    }

    // export the weights and the bias to a txt file
    void exportWeights(std::string filename) const
    {
        std::ofstream outFile(filename, std::ios::app);

        for (int i = 0; i < layerArr.size() - 1; i++)
        {
            for (int j = 0; j < layerArr[i].size(); j++)
                for (int k = 0; k < layerArr[i][j].weights.size(); k++)
                    outFile << i << "/" << j << "/" << k << " " << layerArr[i][j].weights[k] << std::endl;
            outFile << "b-" << i << " " << biasArr[i].output << std::endl;
        }
        outFile << std::endl;
        outFile.close();
    }

    void exportNodeInfo(std::string filename)
    {
        std::ofstream outFile(filename, std::ios::app);
        for (int i = 1; i < layerArr.size(); i++)
        {
            for (int j = 0; j < layerArr[i].size(); j++)
                outFile << i << "/" << j << " " << layerArr[i][j].preActOutput << " " << layerArr[i][j].output << std::endl;
            outFile << "b-" << i - 1 << " " << biasArr[i - 1].output << std::endl;
        }
        outFile << std::endl;
        outFile.close();
    }

    void setLearningRate(double Rate)
    {
        learningRate = Rate;
    }

    // read, train, and output data to a txt file using above functions
    void readandTrain(std::string trainingFile, std::string OutputFileName, int numEpochs, int outputData = 0)
    {
        if (numEpochs < 0)
            throw std::runtime_error("NumEpochs cannot be negative");
        std::ifstream inFile(trainingFile, std::ios::in);
        std::ofstream outFile(OutputFileName, std::ios::out);
        if (!inFile.is_open())
            throw std::runtime_error("TrainingFile could not be opened/found");

        outFile << "Epoch   inputs = targets : outputs | cost function" << std::endl;

        // ignore file header and verify the trainingFile structure matches the neural network structure
        std::string str;
        std::getline(inFile, str);
        str = "";
        std::getline(inFile, str);
        std::istringstream inputString;
        inputString.str(str);
        int inputSize = layerArr[0].size();
        int outputSize = layerArr.back().size();
        for (size_t i{0}; i < (inputSize + outputSize); i++)
        {

            double test;
            inputString >> test;
        }
        if (!inputString || (inputString >> str))
            throw std::runtime_error("Training file does not match the input and output size of the neural net");
        inFile.seekg(0, std::ios::beg);
        std::getline(inFile, str);

        int epoch{0};
        while (epoch + 1 < numEpochs)
        {
            epoch++;
            std::vector<double> inputs;
            std::vector<double> targets;

            // read from file

            for (size_t i{0}; i < inputSize; i++)
            {
                double input;
                inFile >> input;
                inputs.push_back(input);
            }
            for (size_t i{0}; i < outputSize; i++)
            {
                double target;
                inFile >> target;
                targets.push_back(target);
            }

            if (inFile.eof())
                break;

            forwardProp(inputs);
            std::vector<double> outputs = getOutput();

            if (outputData)
            {
                exportWeights("weightData.txt");
                exportNodeInfo("nodeInfo.txt");
            }
            backProp(targets);

            // output to file
            outFile << std::left << std::setw(6) << epoch;

            for (double num : inputs)
                outFile << std::setw(3) << num;
            outFile << "= ";

            for (double num : targets)
                outFile << std::setw(3) << num << " ";

            outFile << "  :  " << std::setprecision(4) << std::fixed;

            for (double num : outputs)
                outFile << std::setw(10) << num << " ";

            outFile << " | " << std::setprecision(7) << costFunction(outputs, targets) << std::endl
                    << std::resetiosflags(std::ios::fixed);
        }
        inFile.close();
        outFile.close();
    }

    void trainMINST(std::string imageMINST, std::string labelsMINST, std::string OutputFileName, int numEpochs, int outputData = 0)
    {
        if (numEpochs < 0)
            throw std::runtime_error("NumEpochs cannot be negative");

        std::ofstream outFile(OutputFileName, std::ios::out);
        if (!outFile.is_open())
            throw std::runtime_error("OutputFile could not be opened/found");

        outFile << "Epoch | targets : outputs | cost function" << std::endl;

        // MNIST dataset configuration
        int inputSize = 784; // 28x28 input images
        int outputSize = 10; // 10 possible digits (0-9)

        std::vector<std::vector<double>> images = readMNISTImages(imageMINST);
        std::vector<int> labels = readMNISTLabels(labelsMINST);

        std::uniform_int_distribution<> disUn(0, images.size() - 1);
     
        int epoch = 0;
        while (epoch < numEpochs)
        {
            epoch++;
            

            int index = disUn(gen);

            int label = labels[index];

            std::vector<double> targets(outputSize, 0.0);
            targets[label] = 1.0;


            forwardProp(images[index]);
            std::vector<double> outputs = getOutput();

            if (outputData)
            {
                exportWeights("weightData.txt");
                exportNodeInfo("nodeInfo.txt");
            }
            backProp(targets);


            if( numEpochs-epoch < 30 ){
                
                outFile << std::endl;
            
                for (int i = 0; i < 784; i++)
                {
                    char pixelChar;
                    if (images[index][i] > 0.5)
                    {
                        pixelChar = '#';
                    }
                    else if (images[index][i] > 0.3)
                    {
                        pixelChar = '.';
                    }
                    else
                    {
                        pixelChar = ' ';
                    }
                    outFile << pixelChar;

                    if ((i + 1) % 28 == 0)
                    {
                        outFile << std::endl;
                    }
                }

                outFile << std::endl; 
            }
            

            outFile << std::left << std::setw(6) << epoch << "| ";

            for (double num : targets)
                outFile << std::setw(3) << num << " ";

            outFile << "  :  " << std::setprecision(4) << std::fixed;

            for (double num : outputs)
                outFile << std::setw(10) << num << " ";

            outFile << " | " << std::setprecision(7) << costFunction(outputs, targets) << std::endl
                    << std::resetiosflags(std::ios::fixed);
        }
        
    }

    void test()
    {
        int inputSize = layerArr[0].size();
        while (true)
        {
            std::cout << "\nInput " << inputSize << " numbers. Type letters to exit: ";
            std::vector<double> inputs;
            double num;
            for (size_t i = 0; i < layerArr[0].size(); i++)
            {
                if (!(std::cin >> num))
                {
                    std::cin.clear();
                    return;
                }
                inputs.push_back(num);
            }

            std::string clear;
            getline(std::cin, clear);
            forwardProp(inputs);
            std::vector<double> outputs = getOutput();

            std::cout << "outputs: ";
            for (double num : outputs)
                std::cout << num << " ";

            std::cout << std::endl;
        }
    }
    virtual double activation(double) = 0;
    virtual double randWeight(int) = 0;
    virtual double derivedActivation(double) = 0;
};

// activation functions
class LeakyRelu : public NeuralNet
{
    double constant{0.1};

public:
    LeakyRelu(const std::vector<int> &map) : NeuralNet(map)
    {
        learningRate = 0.1;
    }
    virtual double activation(double x)
    {
        return ((x > 0) ? x : x * constant);
    }
    virtual double derivedActivation(double x)
    {
        return ((x > 0) ? 1.0 : constant);
    }

    virtual double randWeight(int numOfNodesBefore)
    { // xavier weight intialization
        return dis(gen) * sqrt(2.0 / numOfNodesBefore);
    }
    void setConstant(double x) { constant = x; }
};

class Sigmoid : public NeuralNet
{
public:
    Sigmoid(const std::vector<int> &map) : NeuralNet(map)
    {
        learningRate = 1;
    }
    virtual double activation(double x)
    {
        return (1.0 / (1.0 + exp(-x)));
    }
    virtual double derivedActivation(double x)
    { // xavier weight intialization
        return (activation(x) * (1.0 - activation(x)));
    }

    virtual double randWeight(int numOfNodesBefore)
    {
        return dis(gen) * sqrt(1.0 / numOfNodesBefore);
    }
};

class Tanh : public NeuralNet
{
public:
    Tanh(const std::vector<int> &map) : NeuralNet(map)
    {
        learningRate = 0.1;
    }
    virtual double activation(double x)
    {
        return tanh(x);
    }
    virtual double derivedActivation(double x)
    {
        return 1.0 - pow(tanh(x), 2);
    }

    virtual double randWeight(int numOfNodesBefore)
    { // xavier weight intialization
        return dis(gen) * sqrt(1.0 / numOfNodesBefore);
    }
};

// read MNIST
//  Function to read MNIST images

uint32_t swapEndianness(uint32_t value)
{
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

std::vector<std::vector<double>> readMNISTImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open MNIST image file: " + filename);
    }

    // Read the magic number and other metadata
    uint32_t magicNumber, numImages, numRows, numCols;
    file.read(reinterpret_cast<char *>(&magicNumber), 4);
    file.read(reinterpret_cast<char *>(&numImages), 4);
    file.read(reinterpret_cast<char *>(&numRows), 4);
    file.read(reinterpret_cast<char *>(&numCols), 4);
    magicNumber = swapEndianness(magicNumber);
    numImages = swapEndianness(numImages);
    numRows = swapEndianness(numRows);
    numCols = swapEndianness(numCols);

    // Check if the file format is correct
    if (magicNumber != 0x00000803)
    {
        throw std::runtime_error("Invalid MNIST image file format: " + filename + std::to_string(magicNumber));
    }

    // Read the image data
    std::vector<std::vector<double>> images(numImages, std::vector<double>(numRows * numCols));
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numRows * numCols; ++j)
        {
            unsigned char pixel;
            file.read(reinterpret_cast<char *>(&pixel), 1);
            images[i][j] = static_cast<double>(pixel) / 255.0; // Normalize pixel values to [0, 1]
        }
    }

    return images;
}

// Function to read MNIST labels
std::vector<int> readMNISTLabels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open MNIST label file: " + filename);
    }

    // Read the magic number and other metadata
    int magicNumber, numLabels;
    file.read(reinterpret_cast<char *>(&magicNumber), 4);
    file.read(reinterpret_cast<char *>(&numLabels), 4);
    magicNumber = swapEndianness(magicNumber);
    numLabels = swapEndianness(numLabels);

    // Check if the file format is correct
    if (magicNumber != 0x00000801)
    {
        throw std::runtime_error("Invalid MNIST label file format: " + filename);
    }

    // Read the label data
    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; ++i)
    {
        unsigned char label;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<int>(label);
    }

    return labels;
}



