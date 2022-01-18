import matplotlib.pyplot as plt
import numpy as np
import random
from base import Layer, NeuralNetwork, convert_val_to_class
from data_handler import read_data
from draw import DrawNN

if __name__ == '__main__':
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    data = read_data('data/iris.xlsx')
    data_size = len(data)
    random.shuffle(data)
    training, test = data[:int(data_size*0.7)], data[int(data_size*0.7):]

    # ALPHA = 2.7
    # ARCH = [4, 3, 2, 1]

    ALPHA = 3
    ARCH = [4, 5, 2, 1]

    nn = NeuralNetwork(alpha=ALPHA)
    nn.initialize_layers(*ARCH)
    errors = nn.train(training, test, epoch_num=30)

    x_axis = [e[0] for e in errors]
    train_errors = [e[1] for e in errors]
    test_errors = [e[2] for e in errors]
    conf_matrixes = [e[3] for e in errors]
    conf_matrix = conf_matrixes[-1]

    plt.title("NET ARCH={}   LR={}".format(ARCH.__str__(), ALPHA))
    plt.plot(x_axis, train_errors, label="train", color='tab:blue', )
    plt.plot(x_axis, test_errors, label="test", color='tab:orange', )
    plt.legend()
    plt.show()

    network = DrawNN(ARCH)
    network.draw()

    conf_matrix_as_list = []
    for cl in classes:
        t = []
        for cl2 in classes:
            l = str(cl) + '|||' + str(cl2)
            conf_matrix[l] = conf_matrix.get(l, 0)
            t.append(conf_matrix[l])
        conf_matrix_as_list.append(t)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(
        cellText=conf_matrix_as_list,
        rowLabels=classes,
        colLabels=classes,
        rowColours=["skyblue"] * 3,
        colColours=["skyblue"] * 3,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.1, 0.8, 0.5]
    )

    ax.set_title('Last epoch confusion matrix',
                 fontweight="bold")
    plt.show()


