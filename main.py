import string
from data_process import gen_images
from config import IMAGE_DIR, DATA_DIR


if __name__ == '__main__':
    alphabet = string.ascii_uppercase+string.digits
    gen_images(IMAGE_DIR, alphabet, 4, 5000)
