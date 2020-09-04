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
    x = Conv2D(3, 3, padding='same')(x)
    x = x * 127.5 + div2k_mean
    return Model(inp, x, name="edsr")
