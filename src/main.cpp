#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include "DeepNueralNetwork.h"
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::vector;



int main()
{
    DeepNueralNetwork dnn;
    dnn.ReadFromKeras("/home/nader/workspace/github/CppDNN/example/keras_mnist/w.json");
    MatrixXd input(784,1);
    std::fstream infile("/home/nader/workspace/github/CppDNN/example/keras_mnist/mnist_example");
    for(int i = 0; i < 784; i++)
    {
        double a;  infile>>a;
        input(i,0) = a;
    }
    dnn.Calculate(input);
    cout<<dnn.mOutput<<std::endl;
}
