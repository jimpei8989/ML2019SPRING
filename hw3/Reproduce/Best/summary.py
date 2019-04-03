import sys, os

from keras.models import load_model
from keras.utils import plot_model, print_summary

if __name__ == "__main__":
    lucky_num = 50756711264384381850616619995309447969109689825336919605444730053665222018857 % (2 ** 32)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    modelH5 = sys.argv[1]
    summaryFile = sys.argv[2]
    outputFig = sys.argv[3]

    model = load_model(modelH5)
    with open(summaryFile, 'w') as f:
        print_summary(model, print_fn = lambda x : f.write(x + '\n'))

    plot_model(model, to_file=outputFig)

