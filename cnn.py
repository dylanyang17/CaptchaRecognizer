import torch
from torch import nn
from config import Config


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=.2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=.2),
        )
        self.fc = nn.Linear(432, Config.label_len)

    def forward(self, input_data):
        input_data = input_data.float()
        output = self.cnn2(self.cnn1(input_data.unsqueeze(1)))
        batch_size = output.shape[0]
        output = output.reshape((batch_size, -1))
        output = self.fc(output)
        output = output.reshape((batch_size, Config.captcha_len, len(Config.alphabet)))
        return output
