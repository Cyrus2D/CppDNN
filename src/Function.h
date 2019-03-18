#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
using Eigen::MatrixXd;

enum class Function
{
    Linear, Sigmoid, ReLu, ELu, SoftMax
};
inline Function StringToFunction(std::string function)
{
    if (function.compare("linear") == 0)
    {
        return Function::Linear;
    }
    if (function.compare("relu") == 0)
    {
        return Function::ReLu;
    }
    if (function.compare("elu") == 0)
	{
    	return Function::ELu;
	}
    if (function.compare("sigmoid") == 0)
    {
        return Function::Sigmoid;
    }
    if (function.compare("softmax") == 0)
    {
        return Function::SoftMax;
    }
    
    return Function::Linear; // just for removing warnings
}
static void LinearFunction(MatrixXd & output){
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            output(i, j) = output(i, j);
        }
    }
}
static void ReLuFunction(MatrixXd & output){
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            if( output(i, j) < 0)
                output(i, j) = 0;
            else
                output(i, j) = output(i, j);
        }
    }
}

static void ELuFunction(MatrixXd & output){
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            if( output(i, j) < 0)
                output(i, j) = 0.1*(exp(output(i, j))-1);
            else
                output(i, j) = output(i, j);
        }
    }
}

static void SoftMaxFunction(MatrixXd & output){
    double sum = 0;
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            sum += exp(output(i, j));
        }
    }
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            output(i, j) = exp(output(i, j)) / sum;
        }
    }
}
static void SigmoidFunction(MatrixXd & output){
    for(int i = 0; i < output.rows(); i++){
        for(int j = 0; j < output.cols(); j++){
            output(i, j) = 1.0 / (1.0 + exp(-output(i, j)));
        }
    }
}

