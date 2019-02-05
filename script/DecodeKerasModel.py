import sys
from keras import models

if len(sys.argv) < 3:
    print('usage: python DecodeKerasModel.py input output')
    exit(1)

input = "/home/nader/workspace/github/CppDNN/example/keras_mnist/model.h5"
output = "/home/nader/workspace/github/CppDNN/example/keras_mnist/w.json"
outputFile = open(output, 'w')
model = models.load_model(input)
weights_list = model.get_weights()
outputFile.write("# Layer Numbers: " + str(int(len(weights_list)/2)) + '\n')
for l in range(int(len(weights_list)/2)):
    w = weights_list[l * 2]
    b = weights_list[l * 2 + 1]
    outputFile.write("# Layer Number: {}".format(l) + '\n')
    outputFile.write(model.layers[l].activation.__str__().split(' ')[1] + '\n')
    outputFile.write(str(len(b)) + ' ' + str(len(w)) + '\n')
    outputFile.write("# W" + '\n')
    for x in w:
        for y in x:
            outputFile.write(str(y) + '\n')
    outputFile.write("# B" + '\n')
    for x in b:
        outputFile.write(str(x) + '\n')

