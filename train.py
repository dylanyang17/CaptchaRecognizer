import torch
import pickle
import os
import matplotlib.pyplot as plt

from config import Config

if __name__ == '__main__':
    with open(Config.TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)
        plt.imshow(train_data[0]['image'], cmap='gray')
        plt.show()
        print(train_data[0]['captcha'])
