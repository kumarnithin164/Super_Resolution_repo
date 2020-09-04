from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import PIL
import matplotlib.pyplot as plt


def res_block(inp, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(inp)
    x = Conv2D(filters, 3, padding='same')(x)
    res = Add()([inp, x])
    return res
