import sys, os
import pickle

from keras.models import load_model
from keras.utils import plot_model, print_summary
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    #  os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    modelH5    = sys.argv[1]
    summaryTxt = sys.argv[2]
    outputFig  = sys.argv[3]
    historyPkl = sys.argv[4]
    historyFig = sys.argv[5]

    model = load_model(modelH5)
    with open(summaryTxt, 'w') as f:
        print_summary(model, print_fn = lambda x : f.write(x + '\n'))

    plot_model(model, to_file=outputFig)

    with open(historyPkl, 'rb') as f:
        history = pickle.load(f)

    plt.subplots_adjust(hspace = 0.8)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(history.history['acc'], color="cyan")
    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend(loc='upper left')

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(history.history['loss'], color="orange")
    ax2.set_title("Training Loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend(loc='upper left')

    plt.savefig(historyFig, dpi = 150)

