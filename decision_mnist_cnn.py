'''このプログラムについて
このプログラムは下記に公開されているソースコードと
https://github.com/moritalous/mnist_vs_me
下記記事を参考にしました
https://qiita.com/takus69/items/dd904dfc62372310c46f
'''

import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img
import os
import re
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

def preprocess(image):
    im = image.resize((20, 20))

    im_ret = Image.new('L', (28, 28))
    im_ret.paste(im, (4, 4))

    y_center, x_center =  ndimage.measurements.center_of_mass(np.array(im_ret))
    x_move = x_center - 14
    y_move = y_center - 14
    im_ret = im_ret.transform(size=(28, 28), method=Image.AFFINE,
                              data=(1, 0, x_move, 0, 1, y_move))
    return im_ret

model = load_model('mnist_model.h5')

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
%matplotlib inline
for picture in list_pictures('./tegaki/'):
    %matplotlib inline
    X = []
    img = img_to_array(preprocess(load_img(picture, target_size=(28, 28), grayscale=True)))
    im = Image.open(picture)
    X.append(img)

    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0

    features = model.predict(X)
    
    plt.figure(figsize=(10, 5)) 
    xaxis = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(im))
    height = np.array(features[0])
    plt.subplot(1, 2, 2)
    plt.bar(xaxis, height)
    plt.xticks(xaxis)
    plt.show()
