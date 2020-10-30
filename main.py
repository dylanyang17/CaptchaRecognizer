import string

from data_process import gen_images, rgb2gray, gray_binarization, images2data
from config import IMAGE_DIR, DATA_DIR



if __name__ == '__main__':
    alphabet = string.ascii_uppercase+string.digits
    # gen_images(IMAGE_DIR, alphabet, 4, 5000)
    images2data(IMAGE_DIR, None, os.path.join('test'))
    # image = mpimg.imread('data/images/L8P1.png')
    # image = rgb2gray(image)
    # image = gray_binarization(image)
    # mpimg.imsave('test.png', image)

