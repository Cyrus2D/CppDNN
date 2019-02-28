def NeuralNetworkSaver(ns, layers: list, save_in: str):
    file = open(save_in, "w")
    file.write("# Layer Numbers: " + str(len(layers)))
    file.write('\n')

    for i, layer in enumerate(layers):
        file.write("# Layer Number: " + str(i) + "\n")

        file.write(layer[0] + "\n")

        file.write(str(ns[i+1]) + " " + str(ns[i]) + '\n')

        file.write("# W\n")
        file.write(tensor_to_str(layer[1]))

        file.write("# B\n")
        file.write(tensor_to_str(layer[2]))


def tensor_to_str(tensor):
    out = ""
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            out += str(tensor[i][j]) + "\n"
    return out
