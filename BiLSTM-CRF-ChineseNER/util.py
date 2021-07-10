import matplotlib.pyplot as plt


def draw_train_process(title, data, save_path):
    plt.title(title, fontsize=24)
    plt.plot(data, color='red')
    plt.savefig(save_path)
    plt.show()