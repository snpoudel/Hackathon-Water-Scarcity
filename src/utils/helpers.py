import os


def save_or_create(plt, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))    
    plt.savefig(save_path)
