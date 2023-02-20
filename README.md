# CppDNN

A c++ library to use Keras DNN in c++ programs.

There are two script that can read tensoflow or keras DNN weight from a saved model, and convert them into txt file.
After that you can use the CppDnn to read the txt weight file and use the DNN in c++ programs.

### Install dependency
After that, install Eigen3: https://eigen.tuxfamily.org/dox/index.html
```
sudo apt install libeigen3-dev
```

### Install
```
mkdir build
cd build
cmake ..
make
sudo make install
```


### How to convert a keras model
```
cd script
python DecodeKerasModel.py input-path output-path
```


### How to use the library
There is an example in CppDNN/example/simple_main/main.cpp

```
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
```
