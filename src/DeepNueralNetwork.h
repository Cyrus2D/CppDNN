#include "Layer.h"
#include <vector>

class DeepNueralNetwork{
public:
    MatrixXd mInput;
    MatrixXd mOutput;
    std::vector<Layer> mLayers;
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
