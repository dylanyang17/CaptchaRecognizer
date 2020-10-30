import string
from data_process import gen_images
from config import IMG_DIR, DATA_DIR


if __name__ == '__main__':
    alphabet = string.ascii_uppercase+string.digits
    gen_images(IMG_DIR, alphabet, 4, 5000)
