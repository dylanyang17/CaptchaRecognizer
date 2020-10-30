import os
import logging
import time
import pickle
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from logger import logger
from config import Config
from cnn import CNN


def captcha2label(captcha):
    """
    将 captcha 转为 one-hot 的 label
    :param captcha: str
    :return:
    """
    label = numpy.zeros(Config.captcha_len, dtype=numpy.int64)
    for ind, ch in enumerate(captcha):
        label[ind] = Config.alphabet.index(ch)
    # label = numpy.zeros(Config.label_len)
    # for ind, ch in enumerate(captcha):
    #     label[ind*len(Config.alphabet) + Config.alphabet.index(ch)] = 1
    # label = label.reshape((Config.captcha_len, len(Config.alphabet)))
    return label


class CaptchaDataSet(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        captcha = self.data[item]['captcha']
        return captcha2label(captcha), self.data[item]['image']


def calc_acc(net, device, dataloaders):
    ret = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for dataloader in dataloaders:
            correct = torch.zeros([1]).to(device)
            total = torch.zeros([1]).to(device)
            for i, data in enumerate(dataloader):
                labels, inputs = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                prediction = torch.argmax(outputs, dim=1)
                gt = labels
                y_pred.extend(prediction.cpu().numpy().tolist())
                y_true.extend(gt.cpu().numpy().tolist())
                correct += ((prediction == gt).sum(dim=1) == 4).sum().float()
                # correct += (prediction == gt).sum().float()
                total += len(labels)
            ret.append(float(correct/total))
    return ret


def train(net, device, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader):
    now = time.clock()
    for epoch in range(Config.epoch_num):
        running_loss = 0.0
        cnt = 0
        for i, data in enumerate(train_dataloader):
            labels, inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            cnt += 1
        delta = time.clock() - now
        now = time.clock()
        train_acc, valid_acc, test_acc = calc_acc(net, device, [train_dataloader, valid_dataloader, test_dataloader])
        logger.info('[%d] loss: %.3f  cost: %.3f  train_acc: %.3f  valid_acc: %.3f  test_acc: %.3f' %
                    (epoch + 1, running_loss / cnt, delta, train_acc, valid_acc, test_acc))
        if (epoch) % Config.save_interval == 0:
            torch.save(net, os.path.join(Config.TRAIN_DIR, '%d.model' % epoch))


def train_cnn(train_dataloader, valid_dataloader, test_dataloader):
    """ 开始训练 CNN 模型 """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info('device: %s' % device.__str__())
    net = CNN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=Config.lr)
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
