#pragma once

#include <eigen3/Eigen/Dense>
#include "Function.h"
using Eigen::MatrixXd;
class Layer{
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    MatrixXd mWeight;
    MatrixXd mBios;

    Layer() = delete;
    Layer(const MatrixXd weight, const MatrixXd bios){
        mWeight = weight;
        mBios = bios;
        mFunction = LinearFunction;
    }
    Layer(const MatrixXd weight, const MatrixXd bios, Function function){
        mWeight = weight;
        mBios = bios;
        switch (function) {
        case Function::Linear:
            mFunction = LinearFunction;
            break;
        case Function::ReLu:
            mFunction = ReLuFunction;
            break;
        case Function::ELu:
            mFunction = ELuFunction;
            break;
        case Function::SoftMax:
            mFunction = SoftMaxFunction;
            break;
        case Function::Sigmoid:
            mFunction = SigmoidFunction;
            break;
        }

    }
    void Calculate(){
        mOutput = mWeight * mInput;
        mOutput += mBios;
        mFunction(mOutput);
    }
    void Calculate(MatrixXd input){
        mInput = input;
        Calculate();
    }
    std::function<void(MatrixXd &)> mFunction;
};

