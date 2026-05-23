from keras import backend as K
import tensorflow as tf


#-----------------DICE SIMILARITY COEFFICIENT (DSC)--------------
def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#------------------LOSS FUNCTION - DSC BASED------------------------
def loss_dice(y_true, y_pred):
    smooth = 1.

    y_true = tf.cast(y_true, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    return 1 - dice