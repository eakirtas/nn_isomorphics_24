import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

BASE_COLORS = list(mcolors.BASE_COLORS.keys())[:-1]
TABLEAU_COLORS = list(mcolors.TABLEAU_COLORS)

COLORS = BASE_COLORS + TABLEAU_COLORS


def plot_training(experiments: dict):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2)

    fig.suptitle('Training process', fontsize=16)
    loss_ax.set_title("Loss")
    acc_ax.set_title("Accuracy")

    for i, (key, values) in enumerate(experiments.items()):
        loss_ax.plot(values['loss'], COLORS[i], label=key)
        acc_ax.plot(values['accuracy'], COLORS[i], label=key)

    handles, labels = acc_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()
