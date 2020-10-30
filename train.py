import torch
import pickle
import os
import matplotlib.pyplot as plt

from config import DATA_DIR

if __name__ == '__main__':
    with open(os.path.join(DATA_DIR, 'train_data'), 'rb') as f:
        train_data = pickle.load(f)
        plt.imshow(train_data[0]['image'], cmap='gray')
        plt.show()
        print(train_data[0]['captcha'])
