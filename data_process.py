import os
import pickle
import string
from random import choice, shuffle
from captcha.image import ImageCaptcha
import numpy as np
from numpy import ndarray
from skimage import filters, io, transform
from matplotlib import pyplot as plt

from config import Config


def gen_images(image_dir, alphabet, captcha_len, image_num):
    """
    生成验证码图片，存放到 img_dir 目录下，且图片名同验证码相同
    :param image_dir: str, 图片存放目录
    :param alphabet: list, 验证码的字符取值列表
    :param captcha_len: int, 验证码长度
    :param image_num: int, 生成的图片数目
    :return captchas: list, 生成的验证码列表
    """
    os.makedirs(image_dir, exist_ok=True)
    # 标记是否生成过
    flag = {}
    captchas = []
    for j in range(image_num):
        while True:
            captcha = ''
            for i in range(captcha_len):
                captcha += choice(alphabet)
            if flag.get(captcha, None) is None:
                break
        flag[captcha] = True
        captchas.append(captcha)
        image = ImageCaptcha(fonts=Config.fonts)
        image.generate(captcha)
        image.write(captcha, os.path.join(image_dir, captcha + '.png'))
    return captchas


def images2data(image_dir, captchas, suffix):
    """
    将若干张图片转为便于输入网络的数据格式
    :param image_dir: str, 图片目录
    :param captchas: list, 验证码字符串列表, 也即不带后缀的图片文件名，为 None 时使用 image_dir 下的所有图片
    :param suffix: str, 图片后缀名
    :return 返回列表格式的数据，形如 [{'captcha': ..., 'image': ...}]
    """
    data = []
    if captchas is None:
        captchas = os.listdir(image_dir)
        captchas = list(filter(lambda x: x.split('.')[1] == suffix, captchas))
        captchas = list(map(lambda x: x.split('.')[0], captchas))
    tot_num = len(captchas)
    handled_num = 0
    for captcha in captchas:
        path = os.path.join(image_dir, captcha + '.%s' % suffix)
        image = io.imread(path)/255
        image = transform.resize(image, Config.image_shape)
        image = rgb2gray(image)
        image.astype(np.uint8)
        image = gray_binarization(image)
        data.append({'captcha': captcha, 'image': image})
        handled_num += 1
        print('\rimages2data: %d/%d %.2f%%' % (handled_num, tot_num, handled_num/tot_num*100), end='')
    return data


def rgb2gray(image):
    """
    将 r*c*3 形状的 RGB ndarray 转为灰度图
    :param image: ndarray, RGB 格式的图片
    :return: ndarray，转为 Gray 之后的结果，形状为 r*c
    """
    shape = image.shape
    ret = ndarray((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[i][j] = image[i][j][0]*0.299 + image[i][j][1]*0.587 + image[i][j][2]*0.114
    return ret


def gray_binarization(image):
    """
    将灰度图进行二值化
    :param image: ndarray，Gray 格式的图片
    :return: ndarray, 二值化之后的结果
    """
    thresh = filters.threshold_otsu(image)
    image[image > thresh] = 1
    image[image <= thresh] = -1
    return image


if __name__ == '__main__':
    # Step 0: 定义数据长度
    tot_num = 1000
    valid_num = tot_num//10
    test_num = tot_num//10
    train_num = tot_num - valid_num - test_num

    # Step 1: 生成验证码图片
    # gen_images(Config.IMAGE_DIR, Config.alphabet, 4, tot_num)

    # Step 2: 将验证码图片进行灰度转换、二值处理再合并分别存储到三类数据文件中
    # data = images2data(Config.IMAGE_DIR, None, 'png')
    # shuffle(data)
    # valid_data = data[:valid_num]
    # test_data = data[valid_num:valid_num+test_num]
    # train_data = data[valid_num+test_num:]
    # with open(Config.TRAIN_DATA_PATH, 'wb') as f:
    #     pickle.dump(train_data, f)
    # with open(Config.VALID_DATA_PATH, 'wb') as f:
    #     pickle.dump(valid_data, f)
    # with open(Config.TEST_DATA_PATH, 'wb') as f:
    #     pickle.dump(test_data, f)
