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
        image = self.data[item]['image']
        image[image == 0] = -1
        return captcha2label(captcha), image


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
                prediction = torch.argmax(outputs, dim=2)
                gt = labels
                y_pred.extend(prediction.cpu().numpy().tolist())
                y_true.extend(gt.cpu().numpy().tolist())
                correct += ((prediction == gt).sum(dim=1) == 4).sum().float()
                # correct += (prediction == gt).sum().float()
                total += len(labels)
            ret.append(float(correct / total))
    return ret


def train(net, device, start_epoch, optimizer, train_dataloader, valid_dataloader, test_dataloader):
    now = time.clock()
    for epoch in range(start_epoch, Config.epoch_num):
        running_loss = 0.0
        cnt = 0
        for i, data in enumerate(train_dataloader):
            labels, inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs[:, 0, :], labels[:, 0])
            for j in range(1, Config.captcha_len):
                criterion = nn.CrossEntropyLoss()
                loss = loss + criterion(outputs[:, j, :], labels[:, j])
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            cnt += 1
        delta = time.clock() - now
        now = time.clock()
        train_acc, valid_acc, test_acc = calc_acc(net, device, [train_dataloader, valid_dataloader, test_dataloader])
        logger.info('[%d] loss: %.3f  cost: %.3f  train_acc: %.3f  valid_acc: %.3f  test_acc: %.3f' %
                    (epoch, running_loss / cnt, delta, train_acc, valid_acc, test_acc))
        if epoch % Config.save_interval == 0:
            torch.save(net, os.path.join(Config.TRAIN_DIR, '%d.model' % epoch))


def train_cnn(model_epoch, train_dataloader, valid_dataloader, test_dataloader):
    """
    开始训练 CNN 模型
    :param model_epoch: 要载入的模型的轮数，为 None 时重新训练
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info('device: %s' % device.__str__())
    if model_epoch == 0:
        net = CNN().to(device)
    else:
        net = torch.load(os.path.join(Config.TRAIN_DIR, '%d.model' % model_epoch)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=Config.lr)
    train(net, device, model_epoch+1, optimizer, train_dataloader, valid_dataloader, test_dataloader)


if __name__ == '__main__':
    train_dataset = CaptchaDataSet(Config.TRAIN_DATA_PATH)
    valid_dataset = CaptchaDataSet(Config.VALID_DATA_PATH)
    test_dataset = CaptchaDataSet(Config.TEST_DATA_PATH)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size, pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, num_workers=0, batch_size=Config.batch_size, pin_memory=True)
    train_cnn(0, train_dataloader, valid_dataloader, test_dataloader)
