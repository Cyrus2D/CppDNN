#include <iostream>
#include <Eigen/Dense>
#include <functional>
using Eigen::MatrixXd;
class Layer{
public:
    enum class Function
    {
        Liner, Sigmoid, ReLu, SoftMax
    };
    MatrixXd mInput;
    MatrixXd mOutput;
    MatrixXd mWeight;
    MatrixXd mBios;

    Layer() = delete;
    Layer(const MatrixXd weight, const MatrixXd bios){
        mWeight = weight;
        mBios = bios;
        mFunction = LinerFunction;
    }
    Layer(const MatrixXd weight, const MatrixXd bios, Function function){
        mWeight = weight;
        mBios = bios;
        switch (function) {
        case Function::Liner:
            mFunction = LinerFunction;
            break;
        case Function::ReLu:
            mFunction = ReLuFunction;
            break;
        case Function::SoftMax:
            mFunction = SoftMaxFunction;
            break;
        case Function::Sigmoid:
            mFunction = SigmoidFunction;
            break;
        }

    }
    void CalculateNet(){
        mOutput = mWeight * mInput;
        mFunction(mOutput);
    }
    void CalculateNet(MatrixXd input){
        mInput = input;
        CalculateNet();
    }
    std::function<void(MatrixXd &)> mFunction;
    static void LinerFunction(MatrixXd & output){
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
    static void SoftMaxFunction(MatrixXd & output){
        double min = 0;
        for(int i = 0; i < output.rows(); i++){
            for(int j = 0; j < output.cols(); j++){
                if(min > output(i, j))
                    min = output(i, j);
            }
        }
        double sum = 0;
        for(int i = 0; i < output.rows(); i++){
            for(int j = 0; j < output.cols(); j++){
                sum += (output(i, j) - min);
            }
        }
        for(int i = 0; i < output.rows(); i++){
            for(int j = 0; j < output.cols(); j++){
                output(i, j) = (output(i, j) - min) / sum;
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
};

int main()
{
    MatrixXd w(2,4);
    w.setRandom();
    MatrixXd b(2,1);
    b.setRandom();
    MatrixXd i(4,1);
    i(0,0) = 0;
    i(1,0) = 1;
    i(2,0) = 2;
    i(3,0) = -1;
    Layer a(w, b, Layer::Function::ReLu);
    Layer aa(w, b, Layer::Function::Sigmoid);
    Layer aaa(w, b, Layer::Function::Liner);
    a.CalculateNet(i);
    aa.CalculateNet(i);
    aaa.CalculateNet(i);

    std::cout<<a.mOutput<<std::endl;
    std::cout<<aa.mOutput<<std::endl;
    std::cout<<aaa.mOutput<<std::endl;
}
