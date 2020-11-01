from skimage import io
from matplotlib import pyplot as plt
from captcha.image import ImageCaptcha

if __name__ == '__main__':
    captcha = 'IFIY'
    fonts = ['fonts/hollow.ttf']
    image = ImageCaptcha(fonts=fonts)
    tmp_path = 'tmp.png'
    image.write(captcha, tmp_path)
    image = io.imread(tmp_path) / 255
    plt.imshow(image)
    plt.show()
