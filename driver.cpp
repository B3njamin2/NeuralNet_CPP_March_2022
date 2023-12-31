#include "neuralNet.h"

// The driver is already set up to work with the "trainingALLGates.txt" training file
// check _outputDataAllGates.txt (this can be changed) after running for results
// Check Readme for settings that work with the different training files
int main(){

    // 1* neuralNet layout 

    // the neural net can have multiple hidden layer ex: (2,3,3,3)) means
    // it has (2 input ,2 x 3 nodes hidden layer, 3 output nodes)

    //settings for MiNST dataset 28 x 28 input nodes and 10 output nodes 
    std::vector<int> map = {784, 128, 64, 10};

    //settings for logic Gates 2 input nodes and  4 output nodes
    //std::vector<int> map = {2,8,4};
    
    // 2* choose one activation function 

    LeakyRelu net(map);
    //Sigmoid net(map);
    //Tanh net(map);
    
    // 3* set learning rate
    net.setLearningRate(0.05);

    // 3.5* ONLY for LeakyRelu set constant
    net.setConstant(0.5);

    net.weightIntialization();

    // 4* train network

    
    try{
        // training with MINST dataset
    
        // net.trainMINST("TrainingDataset/MINSTimage", "TrainingDataset/MINSTlabal",num_of_epoch_to_train , 1 or 0 (default 0))  
        // 1 = export weight and node data and prediction results||  0 = only prediction results)

        net.trainMINST("TrainingDataset/train-images.idx3-ubyte", "TrainingDataset/train-labels.idx1-ubyte","Results.txt", 5);

        // training with logic gate data
        // net.readandTrain("TrainingDataset/input_file.txt", "TrainingDataset/output_file_name.txt",num_of_epoch_to_train , 1 or 0 (default 0))

        // net.readandTrain("trainingAllGates.txt","_outputDataAllGates.txt", 3000);
    }
    catch(const std::runtime_error& error){
        std::cout<<"Error : "<<error.what()<<std::endl;
    }
    
    

    // (optional) test neural net with input using terminal and verify the outputs
    // net.test();
    

    /* list of all net. user functions 

        void weightIntialization()
        void forwardProp(const std::vector<double> &inputs);
        void backProp(const std::vector<double> &targets)
        std::vector<double> getOutput()
        double costFunction(const std::vector<double> &outputs,const std::vector<double> &targets) const
        void importWeights(std::string filename)
        void exportWeights(std::string filename )const
        void exportNodeInfo(std::string filename)
        static void setLearningRate(double Rate)
        void readandTrain(std::string trainingFile,std::string OutputFileName,int outputData)
        void test()
    */
}