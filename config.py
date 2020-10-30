import os
import logging
import coloredlogs


class Config:
    DATA_DIR = 'data/'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images/')

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data')
    VALID_DATA_PATH = os.path.join(DATA_DIR, 'valid_data')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data')

    logger = logging.getLogger('logger')
    coloredlogs.install(level='DEBUG', logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')

    batch_size = 32
    lr = 0.001
    momentum = 0.9
