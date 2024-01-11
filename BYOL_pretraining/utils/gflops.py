import tensorflow as tf
import tensorflow as tf
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense, LSTM, TimeDistributed, AveragePooling1D
from transformer import TransformerEncoder, PatchClassEmbedding, Patches
from data import load_mpose, load_kinetics, random_flip, random_noise, one_hot
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras_flops import get_flops


d_models= 192
input_shape = ((30,13))

inputs = tf.keras.Input(shape=input_shape, name='input')
print("this is the input")
print(inputs.shape)
#x = tf.keras.layers.Dense(192)(inputs)
x = tf.keras.layers.Conv1D(filters=192, kernel_size=1, activation='selu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('selu')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
x = tf.keras.layers.Conv1D(filters=192, kernel_size=1, activation='selu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('selu')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
x = tf.keras.layers.Dense(192)(x)
x = PatchClassEmbedding(d_model= d_models)(x)
transformer = TransformerEncoder(d_model= d_models, num_heads= 3, d_ff=768, dropout=0.3, activation = tf.nn.gelu, n_layers= 6)
x = transformer(x)
x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
x = tf.keras.layers.Dense(256)(x)

model = Model(inputs, x,)

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")


