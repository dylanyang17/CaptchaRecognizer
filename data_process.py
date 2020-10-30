import os
from random import choice
from captcha.image import ImageCaptcha


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


def images2data(img_dir):
    """
    将若干张图片转为便于输入网络的数据格式，图片均为 png 格式
    :param img_dir: 图片目录
    :param
    :return:
    """
    pass
