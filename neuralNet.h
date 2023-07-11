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

std::mt19937 gen(time(0));
std::normal_distribution<double> dis(0, 1);

// Each the class node owns the weights connected directly infront
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
