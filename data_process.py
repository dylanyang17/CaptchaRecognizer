import os
from random import choice
from captcha.image import ImageCaptcha
from numpy import ndarray
from skimage import filters


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
        image = ImageCaptcha()
        image.generate(captcha)
        image.write(captcha, os.path.join(image_dir, captcha + '.png'))
    return captchas


def images2data(image_dir, captchas, data_path):
    """
    将若干张图片转为便于输入网络的数据格式，图片均为 png 格式
    :param image_dir: str, 图片目录
    :param captchas: list, 验证码字符串列表, 也即不带后缀的图片文件名
    :param data_path: str, 数据文件路径名
    :return:
    """
    pass


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
    image[image <= thresh] = 0
    return image
