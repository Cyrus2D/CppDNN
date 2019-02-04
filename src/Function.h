#include <Eigen/Dense>
using Eigen::MatrixXd;

enum class Function
{
    Liner, Sigmoid, ReLu, SoftMax
};
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
