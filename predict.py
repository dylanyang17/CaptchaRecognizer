import os
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from captcha.image import ImageCaptcha

from config import Config
from data_process import images2data

net = torch.load(Config.final_net_path)
net.to(torch.device('cpu'))


def run_net(net, data):
    """
    用 data 作为输入返回预测字符串
    :param net: 网络
    :param data: 输入数据
    :return:
    """
    output = net(data)
    prediction = torch.argmax(output, dim=2)
    tmp = prediction.squeeze(0).detach().numpy()
    print(tmp, type(tmp), tmp.dtype)
    result = ''.join(map(lambda x: Config.alphabet[x], tmp))
    return result


def predict(image_path):
    """ 对 image_path 的图片进行预测并返回预测字符串 """
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    data = images2data(dir_name, [base_name.split('.')[0]], base_name.split('.')[1])
    plt.imshow(data[0]['image'])
    plt.show()
    return run_net(net, torch.from_numpy(data[0]['image']).unsqueeze(0))


if __name__ == '__main__':
    image_path = os.path.join(Config.IMAGE_DIR, '8LNR.png')
    image_path = os.path.join('data/small_images', '1KNG.png')
    result = predict(image_path)
    print(result)
