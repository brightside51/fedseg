# LIBRARIES

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from keras.models import Model
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input
from keras.layers import Add, Multiply
from keras.layers import InputSpec
from keras.layers import *
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv1D, Reshape, Conv2DTranspose
from keras.layers import ZeroPadding2D, Add, AveragePooling2D, Dense, Flatten, UpSampling2D

from keras.activations import relu



# FUNCTION FOR RESIDUAL UNET
def residualUNet(summary = False):
    
    ### ResNet34-UNet
    
    inputs = Input((128, 128, 1))

    #Stage0
    conv0 = Conv2D(64, (7,7), padding='same', strides=(2,2))(inputs)
    bn0 = BatchNormalization()(conv0)
    zp01 = ZeroPadding2D(padding=(1,1))(bn0)
    pool0 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(zp01)

    #Stage1 Conv2_x
    #Step1
    conv1 = Conv2D(64, (3,3), padding='same')(pool0)
    bn1 = BatchNormalization()(conv1)
    rl1 = relu(bn1)
    conv1 = Conv2D(64, (3,3), padding='same')(rl1)
    bn1 = BatchNormalization()(conv1)
    conv11 = Add()([bn1, pool0])
    conv11 = relu(conv11)

    #Step2
    conv12 = Conv2D(64, (3,3), padding='same')(conv11)
    bn12 = BatchNormalization()(conv12)
    rl12 = relu(bn12)
    conv12 = Conv2D(64, (3,3), padding='same')(rl12)
    bn12 = BatchNormalization()(conv12)
    conv12 = Add()([bn12, conv11])
    conv12 = relu(conv12)

    #Step3
    conv13 = Conv2D(64, (3,3), padding='same')(conv12)
    bn13 = BatchNormalization()(conv13)
    rl13 = relu(bn13)
    conv13 = Conv2D(64, (3,3), padding='same')(rl13)
    bn13 = BatchNormalization()(conv13)
    conv13 = Add()([bn13, conv12])
    conv13 = relu(conv13)


    #Stage2 conv3_x
    #Step1
    conv2 = Conv2D(128, (3,3), padding='same', strides=(2,2))(conv13)
    bn2 = BatchNormalization()(conv2)
    rl2 = relu(bn2)
    conv2 = Conv2D(128, (3,3), padding='same')(rl2)
    bn2 = BatchNormalization()(conv2)
    skip_c2 = Conv2D(128, (1,1), padding='same', strides=(2,2))(conv13)
    conv21 = Add()([bn2, skip_c2])
    conv21 = relu(conv21)

    #Step2
    conv22 = Conv2D(128, (3,3), padding='same')(conv21)
    bn22 = BatchNormalization()(conv22)
    rl22 = relu(bn22)
    conv22 = Conv2D(128, (3,3), padding='same')(rl22)
    bn22 = BatchNormalization()(conv22)
    conv22 = Add()([bn22, conv21])
    conv22 = relu(conv22)

    #Step3
    conv23 = Conv2D(128, (3,3), padding='same')(conv22)
    bn23 = BatchNormalization()(conv23)
    rl23 = relu(bn23)
    conv23 = Conv2D(128, (3,3), padding='same')(rl23)
    bn23 = BatchNormalization()(conv23)
    conv23 = Add()([bn23, conv22])
    conv23 = relu(conv23)

    #Step4
    conv24 = Conv2D(128, (3,3), padding='same')(conv23)
    bn24 = BatchNormalization()(conv24)
    rl24 = relu(bn24)
    conv24 = Conv2D(128, (3,3), padding='same')(rl24)
    bn24 = BatchNormalization()(conv24)
    conv24 = Add()([bn24, conv23])
    conv24 = relu(conv24)


    #Stage3 conv4_x
    #Step1
    conv3 = Conv2D(256, (3,3), padding='same', strides=(2,2))(conv24)
    bn3 = BatchNormalization()(conv3)
    rl3 = relu(bn3)
    conv3 = Conv2D(256, (3,3), padding='same')(rl3)
    skip_c3 = Conv2D(256, (1,1), padding='same', strides=(2,2))(conv24)
    bn3 = BatchNormalization()(conv3)
    conv31 = Add()([bn3, skip_c3])
    conv31 = relu(conv31)

    #Step2
    conv32 = Conv2D(256, (3,3), padding='same')(conv31)
    bn32 = BatchNormalization()(conv32)
    rl32 = relu(bn32)
    conv32 = Conv2D(256, (3,3), padding='same')(rl32)
    bn32 = BatchNormalization()(conv32)
    conv32 = Add()([bn32, conv31])
    conv32 = relu(conv32)

    #Step3
    conv33 = Conv2D(256, (3,3), padding='same')(conv32)
    bn33 = BatchNormalization()(conv33)
    rl33 = relu(bn33)
    conv33 = Conv2D(256, (3,3), padding='same')(rl33)
    bn33 = BatchNormalization()(conv33)
    conv33 = Add()([bn33, conv32])
    conv33 = relu(conv33)

    #Step4
    conv34 = Conv2D(256, (3,3), padding='same')(conv33)
    bn34 = BatchNormalization()(conv34)
    rl34 = relu(bn34)
    conv34 = Conv2D(256, (3,3), padding='same')(rl34)
    bn34 = BatchNormalization()(conv34)
    conv34 = Add()([bn34, conv33])
    conv34 = relu(conv34)

    #Step5
    conv35 = Conv2D(256, (3,3), padding='same')(conv34)
    bn35 = BatchNormalization()(conv35)
    rl35 = relu(bn35)
    conv35 = Conv2D(256, (3,3), padding='same')(rl35)
    bn35 = BatchNormalization()(conv35)
    conv35 = Add()([bn35, conv34])
    conv35 = relu(conv35)

    #Step6
    conv36 = Conv2D(256, (3,3), padding='same')(conv35)
    bn36 = BatchNormalization()(conv36)
    rl36 = relu(bn36)
    conv36 = Conv2D(256, (3,3), padding='same')(rl36)
    bn36 = BatchNormalization()(conv36)
    conv36 = Add()([bn36, conv35])
    conv36 = relu(conv36)


    #Stage4 conv5_x
    #Step1
    conv4 = Conv2D(512, (3,3), padding='same', strides=(2,2))(conv36)
    bn4 = BatchNormalization()(conv4)
    rl4 = relu(bn4)
    conv4 = Conv2D(512, (3,3), padding='same')(rl4)
    bn4 = BatchNormalization()(conv4)
    skip_c4 = Conv2D(512, (1,1), padding='same', strides=(2,2))(conv36)
    conv41 = Add()([bn4, skip_c4])
    conv41 = relu(conv41)

    #Step2
    conv42 = Conv2D(512, (3,3), padding='same')(conv41)
    bn42 = BatchNormalization()(conv42)
    rl42 = relu(bn42)
    conv42 = Conv2D(512, (3,3), padding='same')(rl42)
    bn42 = BatchNormalization()(conv42)
    conv42 = Add()([bn42, conv41])
    conv42 = relu(conv42)

    #Step3
    conv43 = Conv2D(512, (3,3), padding='same')(conv42)
    bn43 = BatchNormalization()(conv43)
    rl43 = relu(bn43)
    conv43 = Conv2D(512, (3,3), padding='same')(rl43)
    bn43 = BatchNormalization()(conv43)
    conv43 = Add()([bn43, conv43])
    conv43 = relu(conv43)



    ## UPSAMPLING
    up6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv43), conv36], axis=3)
    conv6 = Conv2D(256, (3,3), padding='same')(up6)
    bn6 = BatchNormalization()(conv6)
    rl6 = relu(bn6)
    conv6 = Conv2D(256, (3,3), padding='same')(rl6)
    bn6 = BatchNormalization()(conv6)
    rl6 = relu(bn6)



    up7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(rl6), conv24], axis=3)
    conv7 = Conv2D(128, (3,3), padding='same')(up7)
    bn7 = BatchNormalization()(conv7)
    rl7 = relu(bn7)
    conv7 = Conv2D(128, (3,3), padding='same')(rl7)
    bn7 = BatchNormalization()(conv7)
    rl7 = relu(bn7)



    up8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(rl7), conv13], axis=3)
    conv8 = Conv2D(64, (3,3), padding='same')(up8)
    bn8 = BatchNormalization()(conv8)
    rl8 = relu(bn8)
    conv8 = Conv2D(64, (3,3), padding='same')(rl8)
    bn8 = BatchNormalization()(conv8)
    rl8 = relu(bn8)



    up9 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(rl8), bn0], axis=3)
    conv9_0 = Conv2D(64, (7, 7), padding='same')(up9)
    bn9 = BatchNormalization()(conv9_0)
    rl9 = relu(bn9)
    conv9 = Conv2D(64, (7, 7), padding='same')(rl9)
    bn9 = BatchNormalization()(conv9)
    rl9 = relu(bn9)


    up10 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(rl9)
    conv10 = Conv2D(16, (3, 3), padding='same')(up10)
    bn10 = BatchNormalization()(conv10)
    rl10 = relu(bn10)
    conv10 = Conv2D(16, (3, 3), padding='same')(rl10)
    bn10 = BatchNormalization()(conv10)
    rl10 = relu(bn10)

    conv_11 = Conv2D(1, (1, 1), activation='sigmoid')(rl10)

    resnet = Model(inputs=[inputs], outputs=[conv_11], name='resNet')

    if summary:
        resnet.summary()

    return resnet
