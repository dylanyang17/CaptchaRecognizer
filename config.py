import os


class Config:
    DATA_DIR = 'data/'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images/')

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid_data')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data')

    batch_size = 32
