
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf

def conv3d_2(data,channels,kernel_size):
    data = Conv3D(channels, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    data = Conv3D(channels, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    return data
def conv3d_2drop(data,channels,kernel_size):
    data = Conv3D(channels, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    data = Dropout(0.5)(data)
    data = Conv3D(channels, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    return data
def Up3D(data,channels,kernel_size):
    data=UpSampling3D(size=(2, 2,2))(data)
    data = Conv3D(channels, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    return data
def conv_out3D(data,kernel_size):
    data = Conv3D(16, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    data = Conv3D(8, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(data)
    return data

def unet(pretrained_weights=None, input_size=(None, None, None,1)):
    inputs = Input(input_size)
    #下采样
    size=5
    down1=conv3d_2(inputs,32,size)
    down2 = MaxPooling3D(pool_size=(2, 2,2))(down1)
    down2 = conv3d_2(down2, 64,size)
    down3 = MaxPooling3D(pool_size=(2, 2,2))(down2)
    down3 = conv3d_2(down3, 128,size)
    down4 = MaxPooling3D(pool_size=(2, 2,2))(down3)
    down4 = conv3d_2drop(down4,256,size)
    down5 = MaxPooling3D(pool_size=(2, 2,2))(down4)
    down5 = conv3d_2drop(down5,512,size)

    #上采样得到大标签
    up4_big = Up3D(down5, 512, size)
    up4_big = concatenate([down4, up4_big], axis=-1)
    up4_big = conv3d_2drop(up4_big, 256,size)
    up3_big   = Up3D(up4_big,256,size)
    up3_big = concatenate([down3, up3_big], axis=-1)
    up3_big = conv3d_2(up3_big,128,size)
    up2_big   = Up3D(up3_big,128,size)
    up2_big = concatenate([down2, up2_big], axis=-1)
    up2_big = conv3d_2(up2_big, 64,size)
    up1_big   = Up3D(up2_big,64,size)
    up1_big = concatenate([down1, up1_big], axis=-1)
    up1_big = conv3d_2(up1_big, 32,size)
    up1_big_out=conv_out3D(up1_big,size)
    up1_big_out = Conv3D(2, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal',name="up1_big_out")(up1_big_out)



    model = Model(inputs=[inputs], outputs=[up1_big_out])

    model.compile(optimizer=Adam(lr=0.00001), loss=[loss1], metrics=['accuracy'])

    # model.summary()

    # if(pretrained_weights):
    # model.load_weights(pretrained_weights)

    return model

def loss1(y_true, y_pred):
    BCE=tf.keras.losses.binary_crossentropy(y_true=y_true,y_pred=y_pred)
    DC= dice_coef_loss(y_true=y_true, y_pred=y_pred)
    return DC+BCE

def loss2(y_true, y_pred):
    BCE=tf.keras.losses.binary_crossentropy(y_true=y_true,y_pred=y_pred)
    DC= dice_coef_loss(y_true=y_true, y_pred=y_pred)
    return DC+BCE

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)