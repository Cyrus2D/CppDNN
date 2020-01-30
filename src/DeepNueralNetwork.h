#pragma once


#include "Layer.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

class DeepNueralNetwork{
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    std::vector<Layer> mLayers;
    void AddLayer(const Layer layer)
    {
        mLayers.push_back(layer);
    }
    void Calculate()
    {
        MatrixXd * out = &mInput;
        for(size_t l = 0; l < mLayers.size(); l++)
        {
//             std::cout<<"calc l"<<std::endl;
             mLayers[l].Calculate(*out);
             out = &(mLayers[l].mOutput);
//             std::cout<<"out"<<(*out)(0,0)<<","<<(*out)(1,0)<<std::endl;
        }
        mOutput = (*out);
    }

    void Calculate(MatrixXd input)
    {
        mInput = input;
        Calculate();
    }

    bool ReadFromKeras(std::string file)
    {
        std::fstream infile(file);
        std::string line;
        std::getline(infile, line);
        int layerSize = std::stod(line.substr(line.find_last_of(" "), line.size()));
        std::cout << "layerSize: " << layerSize << std::endl;
        for (int l = 0; l < layerSize; l++)
        {
            std::getline(infile, line);
            std::string activation;
            std::getline(infile, line);
            std::istringstream acc(line);
            acc >> activation;
            std::getline(infile, line);
            std::istringstream iss(line);
            int mSize, nSize;
            iss >> mSize >> nSize;
            std::getline(infile, line);
            MatrixXd W(mSize, nSize);
            for (int n = 0; n < nSize; n++)
            {
                for (int m = 0; m < mSize; m++)
                {
                    std::getline(infile, line);
                    std::istringstream iss(line);
                    double w; iss >> w;
                    W(m, n) = w;
                }
            }
            std::getline(infile, line);
            MatrixXd B(mSize, 1);
            for (int m = 0; m < mSize; m++)
            {
                std::getline(infile, line);
                std::istringstream iss(line);
                double b; iss >> b;
                B(m, 0) = b;
            }
            AddLayer(Layer(W, B, StringToFunction(activation)));
        }
        return true; //for remove warning
    }

    bool ReadFromTensorFlow(std::string file)
    {
		ReadFromKeras(file);
		return true; //for remove warning
    }
};

