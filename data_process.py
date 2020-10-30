import os
from random import choice
from captcha.image import ImageCaptcha


def gen_images(img_dir, alphabet, captcha_len, img_num):
    """
    生成验证码图片，存放到 img_dir 目录下，且图片名同验证码相同
    :param img_dir: str, 图片存放目录
    :param alphabet: list, 验证码的字符取值列表
    :param captcha_len: int, 验证码长度
    :param img_num: int, 生成的图片数目
    :return captchas: list, 生成的验证码列表
    """
    os.makedirs(img_dir, exist_ok=True)
    # 标记是否生成过
    flag = {}
    captchas = []
    for j in range(img_num):
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
        image.write(captcha, os.path.join(img_dir, captcha + '.png'))
    return captchas

