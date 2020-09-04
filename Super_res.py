from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import PIL
import matplotlib.pyplot as plt


div2k_mean = np.array([0.4488, 0.4371, 0.4040]) * 255     # pre-computed div2k mean


