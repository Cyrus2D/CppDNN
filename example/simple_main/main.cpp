#include <iostream>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <vector>
//#include "DeepNueralNetwork.h"
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::vector;

#include <CppDNN/DeepNueralNetwork.h>


int main()
{
    DeepNueralNetwork dnn;
    dnn.ReadFromKeras("/home/nader/workspace/github/CppDNN/example/keras_simple/simple.txt");
    MatrixXd input(5,1);
    input(0,0) = 1;
    input(1,0) = 2;
    input(2,0) = 1;
    input(3,0) = 2;
    input(4,0) = 1;
    dnn.Calculate(input);
    cout<<dnn.mOutput<<std::endl;
}
