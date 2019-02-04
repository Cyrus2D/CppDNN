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
    Layer a(w1, b1, Function::ReLu);
    Layer aa(w2, b2, Function::Sigmoid);
    Layer aaa(w3, b3, Function::Liner);

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
