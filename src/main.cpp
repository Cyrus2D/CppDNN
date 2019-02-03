#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <vector>
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::vector;
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
    void Calculate(){
        mOutput = mWeight * mInput;
        mFunction(mOutput);
    }
    void Calculate(MatrixXd input){
        mInput = input;
        Calculate();
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
class DeepNueralNetwork{
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    vector<Layer> mLayers;
    void AddLayer(const Layer layer)
    {
        mLayers.push_back(layer);
    }
    void Calculate(){
        MatrixXd * out = &mInput;
        for(size_t l = 0; l < mLayers.size(); l++)
        {
             mLayers[l].Calculate(*out);
             out = &(mLayers[l].mOutput);
        }
        mOutput = (*out);
    }
    void Calculate(MatrixXd input){
        mInput = input;
        Calculate();
    }
};

int main()
{
    MatrixXd w1(2,4);
    w1.setRandom();
    MatrixXd b1(2,1);
    b1.setRandom();

    MatrixXd w2(3,2);
    w2.setRandom();
    MatrixXd b2(3,1);
    b2.setRandom();

    MatrixXd w3(2,3);
    w3.setRandom();
    MatrixXd b3(2,1);
    b3.setRandom();

    MatrixXd i(4,1);
    i(0,0) = 0;
    i(1,0) = 1;
    i(2,0) = 2;
    i(3,0) = -1;
    Layer a(w1, b1, Layer::Function::ReLu);
    Layer aa(w2, b2, Layer::Function::Sigmoid);
    Layer aaa(w3, b3, Layer::Function::Liner);

    DeepNueralNetwork dnn;
    dnn.AddLayer(a);
    dnn.AddLayer(aa);
    dnn.AddLayer(aaa);
    dnn.Calculate(i);
    cout<<"i:"<<i<<endl;
    cout<<"w1:"<<w1<<endl;
    cout<<"w2:"<<w2<<endl;
    cout<<"w3:"<<w3<<endl;
    cout<<dnn.mOutput<<std::endl;
}
