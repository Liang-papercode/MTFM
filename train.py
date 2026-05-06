from numpy.random import seed
seed(12345)

import matplotlib.pyplot as plt
from Data import DataGenerator
from model import *
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint
def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':4,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}

  data  = "E:\正负斜率断层识别\二维三维融合网络\训练数据\标签\数据/"
  data2 = "E:\正负斜率断层识别\二维三维融合网络\训练数据\二维\I_X融合注意力并集直接相加/"
  per1  = "E:\正负斜率断层识别\二维三维融合网络\训练数据\标签\正斜率/"
  per2  = "E:\正负斜率断层识别\二维三维融合网络\训练数据\标签\负斜率/"

  #train_ID = range(200)
  data_IDs = list(range(0, 200))  # 创建一个包含所有样本编号的列表
  # 使用 sklearn 的 train_test_split 随机划分数据集
  train_ID, valid_ID = train_test_split(data_IDs, test_size=0.1, random_state=42)  # 20%的数据作为验证集

  valid_generator = DataGenerator(per1=per1,per2=per2,data=data,
                                  data2=data2,
                                  data_IDs=valid_ID,**params)
  train_generator = DataGenerator(per1=per1,per2=per2,data=data,
                                  data2=data2,
                                  data_IDs=train_ID,**params)
  #model = load_model("unet.hdf5", custom_objects={"loss1": loss1})
  model = unet(input_size=(None,None,None,1))
  model.summary()
  model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

  #history = model.fit_generator(generator=train_generator,epochs=100, callbacks=[model_checkpoint], verbose=1)

  history = model.fit(
      train_generator,
      validation_data=valid_generator,
      epochs=10,
      callbacks=[model_checkpoint],
      verbose=1)
  # 绘制损失曲线
  plt.figure(figsize=(8, 5))
  plt.plot(history.history['loss'], label='Training loss')
  plt.plot(history.history['val_loss'], label='Validation loss')
  plt.title('Loss Curve')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # 绘制精度曲线
  plt.figure(figsize=(8, 5))
  plt.plot(history.history['accuracy'], label='Training accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation accuracy')
  plt.title('Accuracy curve')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  # 显示图像
  plt.show()

  # 创建DataFrame,将损失和精度数据保存到文件
  data = {
      'epoch': range(1, len(history.history['loss']) + 1),
      'Training loss': history.history['loss'],
      'Validation loss': history.history['val_loss'],
      'Training accuracy': history.history['accuracy'],
      'Validation accuracy': history.history['val_accuracy']
  }
  df = pd.DataFrame(data)

  # 保存到CSV文件
  df.to_csv('./3Dunet.csv', index=False)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



if __name__ == '__main__':
    main()



