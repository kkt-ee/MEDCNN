import tensorflow as tf
import tensorflow.keras.backend as K

# # tf 2.15
# def dice_coef(y_true, y_pred, smooth=0):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return dice

# tf 2.16
# import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred, smooth=0):
    # print(y_true.dtype, y_pred.dtype)
    y_true = tf.cast(y_true, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice