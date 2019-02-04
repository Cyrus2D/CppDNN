from keras import models, layers


model = models.load_model("Classification/model.h5")
weights_list = model.get_weights()
print("Layer Numbers:", int(len(weights_list)/2))
for l in range(int(len(weights_list)/2)):
    w = weights_list[l * 2]
    b = weights_list[l * 2 + 1]
    print("# Layer Number: {}".format(l))
    print(model.layers[l].activation.__str__())
    print(len(b), len(w))
    print("# W")
    for x in w:
        for y in x:
            print(y)
    print("# B")
    for x in b:
        print(x)

