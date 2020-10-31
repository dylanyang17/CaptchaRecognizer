import os
import string


class Config:
    DATA_DIR = 'data/'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images/')
    TRAIN_DIR = 'train/'
    MODELS_DIR = 'models/'

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid_data')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data')

    # 目前的验证码：60*160，且只含有四位字符，每个字符为数字或大写字母
    image_shape = (60, 160)
    captcha_len = 4
    alphabet = string.ascii_uppercase + string.digits
    label_len = len(alphabet) * captcha_len
    final_net_path = os.path.join(MODELS_DIR, '560.model')

    batch_size = 128
    lr = 0.0001
    epoch_num = 10000
    save_interval = 20
