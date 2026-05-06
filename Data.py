import numpy as np
import tensorflow as tf

from tensorflow import keras
import random
from tensorflow.keras.utils import to_categorical
import numpy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import LeakyReLU


class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,data,data2,per1,per2,data_IDs, batch_size=1, dim=(300,350),
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.data = data
    self.data2 = data2
    self.per1 = per1
    self.per2 = per2
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):

    per1 = np.fromfile(self.per1+str(data_IDs_temp[0])+'.dat',dtype=np.float32).reshape(128,128,128,1)
    per2 = np.fromfile(self.per2 + str(data_IDs_temp[0]) + '.dat', dtype=np.float32).reshape(128,128,128,1)
    per  = concatenate([per1,per2],axis=-1)
    data = np.fromfile(self.data + str(data_IDs_temp[0]) + '.dat', dtype=np.float32).reshape(128,128,128)
    data2 = np.fromfile(self.data2 + str(data_IDs_temp[0]) + '.dat', dtype=np.float32).reshape(128,128,128)
    data = data*tf.sigmoid(data2)
    data = (data-np.min(data))/(np.max(data)-np.min(data))


    per = np.reshape(per,(1,128,128,128,2))
    data = np.reshape(data,(1,128,128,128,1))

    return data, {"up1_big_out":per}


if __name__ == '__main__':
    DataGenerator()