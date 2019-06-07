import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.switch_backend('agg')

    # Problem 2
    numParas = [39911, 94471, 166279]
    valScore = [0.555904, 0.606061, 0.621212]
    pubScore = [0.54388, 0.60295, 0.60741]
    priScore = [0.55252, 0.60016, 0.60284]

    fig, ax = plt.subplots()
    ax.plot(numParas, valScore, c = '#ff8c00', label = 'val_accu')
    ax.plot(numParas, pubScore, c = '#9dd7ef', label = 'pub_accu')
    ax.plot(numParas, priScore, c = '#90ee90', label = 'pri_accu')

    ax.set_title('Problem 2')
    ax.set_xlabel('#parameters')
    ax.set_ylabel('Accuracy')
    ax.legend()

    fig.savefig('Prob2.png', dpi = 150)


    # Problem 2
    numParas = [52567, 157223, 419047]
    valScore = [0.428596, 0.599443, 0.616066]
    pubScore = [0.42128, 0.58010, 0.60824]
    priScore = [0.43689, 0.59152, 0.60741]

    fig, ax = plt.subplots()
    ax.plot(numParas, valScore, c = '#ff8c00', label = 'val_accu')
    ax.plot(numParas, pubScore, c = '#9dd7ef', label = 'pub_accu')
    ax.plot(numParas, priScore, c = '#90ee90', label = 'pri_accu')

    ax.set_title('Problem 3')
    ax.set_xlabel('#parameters')
    ax.set_ylabel('Accuracy')
    ax.legend()

    fig.savefig('Prob3.png', dpi = 150)


if __name__ == "__main__":
    main()

