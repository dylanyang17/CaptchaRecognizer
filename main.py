import string
import matplotlib.image as mpimg
from data_process import gen_images, rgb2gray, gray_binarization
from config import IMAGE_DIR, DATA_DIR



if __name__ == '__main__':
    alphabet = string.ascii_uppercase+string.digits
    # gen_images(IMAGE_DIR, alphabet, 4, 5000)
    image = mpimg.imread('data/images/0HP6.png')
    image = rgb2gray(image)
    image = gray_binarization(image)
    mpimg.imsave('test.png', image)

