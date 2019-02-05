from keras import models
from keras import layers
from numpy import array

mnistfile = open('mnist_example', 'r')
mnistdata = mnistfile.read()
mnistdata = mnistdata.splitlines()[0].split(' ')
mnistdataf = []
for m in mnistdata:
    mnistdataf.append(float(m))
mnistdata = array(mnistdataf)
mnistdata = mnistdata.reshape((1, 784))

mnistdata = [1, 2, 1, 2, 1]
mnistdata = array(mnistdata)
mnistdata = mnistdata.reshape((1, 5))
network = models.Sequential()
network.add(layers.Dense(8, input_shape=(5,)))
network.add(layers.Dense(5, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print(network.predict(mnistdata))

network.save('simple.h5')
