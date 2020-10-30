import torch
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from config import Config
from cnn import CNN


class CaptchaDataSet(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def train(net, device, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader):
    pass


def train_cnn(train_dataloader, valid_dataloader, test_dataloader):
    """ 开始训练 CNN 模型 """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = CNN().to(device)
    optimizer = optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum)
    criterion = nn.CrossEntropyLoss()
    train(net, device, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader)


if __name__ == '__main__':
    train_dataset = CaptchaDataSet(Config.TRAIN_DATA_PATH)
    valid_dataset = CaptchaDataSet(Config.VALID_DATA_PATH)
    test_dataset = CaptchaDataSet(Config.TEST_DATA_PATH)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size)
    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size)
    train_cnn(train_dataloader, valid_dataloader, test_dataloader)
