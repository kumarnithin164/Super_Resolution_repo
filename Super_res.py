from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import PIL
import matplotlib.pyplot as plt


div2k_mean = np.array([0.4488, 0.4371, 0.4040]) * 255     # pre-computed div2k mean


def edsr():
    scale = 4
    num_filters=64
    num_res_blocks=16
    inp = Input(shape=(None, None, 3))     # produces tensor |||lar to placeholder
    norm = (inp - div2k_mean) / 127.5
    x = res = Conv2D(num_filters, 3, padding='same')(norm)
    for i in range(num_res_blocks):
        res = res_block(res, num_filters)
    after_res = Conv2D(num_filters, 3, padding='same')(res)
    x = Add()([after_res, x])
    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    x = x * 127.5 + div2k_mean
    return Model(inp, x, name="edsr")


def res_block(inp, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(inp)
    x = Conv2D(filters, 3, padding='same')(x)
    res = Add()([inp, x])
    return res


def upsample(x, scale, num_filters):
    factor = 2
    x = Conv2D(num_filters * (factor ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, factor)
    x = Conv2D(num_filters * (factor ** 2), 3, padding='same')(x)
    upsampled = tf.nn.depth_to_space(x, factor)
    return upsampled


edsr_fine_tuned = edsr()
edsr_fine_tuned.load_weights('weights-edsr-16-x4-fine-tuned.h5')


for i in range(1,3):
    low_res = np.array(PIL.Image.open('drive/My Drive/test' + str(i) + '.png'))
    plt.figure(figsize=(18, 18))
    plt.subplot(2, 2, 3)
    plt.imshow(low_res)
    low_res = tf.cast(tf.expand_dims(low_res, axis=0), tf.float32)
    high_res = edsr_fine_tuned(low_res)
    high_res = tf.clip_by_value(high_res, 0, 255)
    high_res = tf.round(high_res)
    high_res = tf.cast(high_res, tf.uint8)

    plt.subplot(2, 2, 4)
    plt.imshow(high_res[0])
