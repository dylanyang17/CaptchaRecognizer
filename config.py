import os
import string


class Config:
    DATA_SIZE = 'big'
    DATA_DIR = 'data/'
    IMAGE_DIR = os.path.join(DATA_DIR, '%s_images/' % DATA_SIZE)
    TRAIN_DIR = 'train/'
    MODELS_DIR = 'models/'
    FONTS_DIR = 'fonts/'

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, '%s_train_data' % DATA_SIZE)
    VALID_DATA_PATH = os.path.join(DATA_DIR, '%s_valid_data' % DATA_SIZE)
    TEST_DATA_PATH = os.path.join(DATA_DIR, '%s_test_data' % DATA_SIZE)

    # 目前的验证码：60*160，且只含有四位字符，每个字符为数字或大写字母，字体为 fonts
    image_shape = (60, 160)
    fonts = ['Times New Roman.ttf']
    for i in range(len(fonts)):
        fonts[i] = os.path.join(FONTS_DIR, fonts[i])
    captcha_len = 4
    alphabet = string.ascii_uppercase + string.digits
    label_len = len(alphabet) * captcha_len
    final_net_path = os.path.join(MODELS_DIR, '200.model')

    batch_size = 128
    lr = 0.001
    epoch_num = 10000
    save_interval = 20
