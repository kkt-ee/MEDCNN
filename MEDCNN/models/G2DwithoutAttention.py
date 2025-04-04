""" Multiresolution Encoder-Decoder Convolutional Neural Network (MEDCNN) without attentions
    Copyright (C) 2025 Kishore Kumar Tarafdar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""


# # include ../dirx 
# mylibpath = [
#     # '/home/kishor/src/FastDWTConvLayers',
#     '/home/kishor/src/MRSegmentation/Attentions19102023'
#     #'/home/k/PLAYGROUND10GB/SKULSTRIPpaper__'
#     ]
# import sys
# [sys.path.insert(0,_) for _ in mylibpath]
# del mylibpath

from TFDWT.DWTIDWT2Dv1 import DWT2D, IDWT2D
from tensorflow.keras.layers import Concatenate

import tensorflow as tf


#%% Pooling layers
# @keras.saving.register_keras_serializable()
class Pooling(tf.keras.layers.Layer):
    """DWT Pooling Layer: keep Low freq band only
   
    MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network
    Copyright (C) 2025 Kishore Kumar Tarafdar
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """
    def __init__(self, Ψ='haar', **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.Ψ = Ψ 

    #  def build(self, input_shape):
    #     self.num_channels = input_shape[-1]
    #     super(Pooling, self).build(input_shape) 

    def call(self, inputs):
        """inputs -> wave0 -> wave1 -> wave2 -> wave0_cap(inverse)"""
        chans = inputs.shape[3]
        wave0 = inputs #L0
        # wave1 = WaveTFFactory.build(self.Ψ)(wave0) #L1
        wave1 = DWT2D(self.Ψ)(wave0)
        return wave1[:,:,:,:chans]
        
#%%
configs = {
    # 'minimal': (2, 2, 2, 2, 2),
    'minimal2': (2, 4, 8, 16, 32),

    # 'config242': (2, 2, 2, 2, 4),
    # 'config282': (2, 2, 2, 4, 4),
    # 'config242': (2, 2, 4, 4, 4),
    # 'config282': (2, 4, 4, 4, 4),
    # 'config282': (2, 4, 4, 4, 8),
    # 'config282': (2, 4, 4, 8, 8),
    # 'config2F2': (2, 4, 8, 8, 8),
    # 'config282': (2, 4, 8, 16, 16),



    # 'config242': (4, 4, 4, 4, 4),
    # 'config282': (4, 4, 4, 4, 8),
    # 'config282': (4, 4, 4, 8, 8),
    # 'config282': (4, 4, 8, 8, 8),
    # 'config242': (4, 8, 8, 8, 8),
    # 'config282': (8, 8, 8, 8, 8),
    # 'config282': (8, 8, 8, 8, 16),
    # 'config2F2': (8, 8, 8, 16, 16),
    # 'config2F2': (8, 8, 16, 16, 16),
    # 'config2F2': (8, 8, 16, 16, 16),
    # 'config2F2': (8, 16, 16, 16, 16),
    # 'config2F2': (16, 16, 16, 16, 16),
    # 'config2F2': (2, 4, 8, 16, 32),
    # 'config2F2': (2, 4, 8, 16, 16),
    # 'config2F2': (2, 4, 8, 16, 16),
    
    
    # 'config4': (4, 4, 4, 4, 4),
    # 'config484': (4, 8, 8, 8, 4),
    # 'config4F4': (4, 16, 16, 16, 4),
    # 'config8': (8, 8, 8, 8, 8),
    # 'default': (8, 16, 16, 16, 8),
    # 'default2': (8, 16, 32, 16, 8)
}


#%%
def conv2D(f, x, activation='relu'):
    """Double conv layer"""
    x = tf.keras.layers.Conv2D(f, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(f, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    # x = CBAM(reduction_ratio=2)(x)
    return x


N_CLASSES = 2 # Mutliclass
N_INPUT_CHANNELS = 1 # tens
# wave = 'haar'

#from keras.engine.training import h5py
# def dwtunet(dataset='IBSR', config=configs['config82']):
def Gφψ(
    Ψ='haar', 
    n_classes=N_CLASSES, 
    n_input_channels=N_INPUT_CHANNELS, 
    scale=1, 
    input_shape =(256, 256, 1),
    # dataset='IBSR', 
    config=configs['minimal2'],
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    residual=False,
    compile=True):
    """ Residual MEDCNN without attention for binary semantic segmentation
   
    Multiresolution features:
        1. DWT level 4 decompostion of input matrix
        2. Wavelet pooling

    MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network
    Copyright (C) 2025 Kishore Kumar Tarafdar
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """

    activation, activation2, activation3 = 'relu', 'relu', tf.keras.activations.mish
    # tfa.activations.mish


    # input_shape =(256, 256, 1)
    inputs = tf.keras.layers.Input(input_shape)
    # s  = tf.keras.layers.Resizing(
    #                 height=256,
    #                 width=256,
    #                 interpolation='bilinear',
    #                 crop_to_aspect_ratio=False
    #             )(inputs)    
    
    s = inputs
    
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(s)
    # normalize
    s = tf.keras.layers.Lambda(lambda x: tf.where(
        tf.reduce_sum(x) !=0,
        (((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)) - (tf.reduce_min((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)))) 
       / ((tf.reduce_max((x - tf.reduce_mean(x)) / tf.math.reduce_std(x))) - (tf.reduce_min((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)))), 
        x
        ))(s)
 
    
    l = DWT2D(wave=Ψ)(s)
    l1 = l[:,:,:,:1]
    h1 = l[:,:,:,1:]
    # print(l1.shape)
    #
    l = DWT2D(wave=Ψ)(l1)
    l2 = l[:,:,:,:1]
    h2 = l[:,:,:,1:]
    #
    l = DWT2D(wave=Ψ)(l2)
    l3 = l[:,:,:,:1]
    h3 = l[:,:,:,1:]
    #
    l = DWT2D(wave=Ψ)(l3)
    l4 = l[:,:,:,:1]
    h4 = l[:,:,:,1:]
    #
    
    # f, a, b, c, d = 8, 16, 16, 16, 8
    # f, a, b, c, d = config
    a, b, c, d , e = config


    #LP process
    # xl1_ = conv2D(f, l1)
    xl1_ = conv2D(a, l1)
    # xl1 = tf.keras.layers.MaxPooling2D((2, 2))(xl1_)
    xl1 = Pooling()(xl1_)    

    # xl2 = conv2D(f, l2)
    # xl2 = l2
    xl2 = Concatenate()([xl1, l2])
    # xl2 = tf.keras.layers.Conv2D(a, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(xl2)
    # xl2_ = conv2D(a, xl2)
    xl2_ = conv2D(b, xl2)
    xl2 = Pooling()(xl2_)   

    # xl3 = conv2D(f, l3)
    # xl3 = l3
    xl3 = Concatenate()([xl2, l3])
    # xl3 = tf.keras.layers.Conv2D(b, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(xl3)
    # xl3_ = conv2D(b, xl3)
    xl3_ = conv2D(c, xl3)
    xl3 = Pooling()(xl3_)
    
    # xl4 = conv2D(f, l4)
    # xl4 = l4
    xl4 = Concatenate()([xl3, l4])
    # xl4 = tf.keras.layers.Conv2D(c, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(xl4)
    # xl4_ = conv2D(c, xl4)
    xl4_ = conv2D(d, xl4)
    xl4 = Pooling()(xl4_)


    xlat = xl4
    # e = f
    # floor------------
    xlat = conv2D(e, xlat)
    #-----------------    


    ## Decoder /Synthesis -----------------

    xl4u_ = tf.keras.layers.Conv2DTranspose(d, (2, 2), strides=(2, 2), padding='same')(xlat)
    xl4u_ = Concatenate()([xl4_, xl4u_])
    # xl4u_ = conv2D(c, xl4u_)
    xl4u_ = conv2D(d, xl4u_)
    xl4u = xl4u_    


    #xl4u_ #up
    #xl4u_ pass
    
    xl3u_ = tf.keras.layers.Conv2DTranspose(c, (2, 2), strides=(2, 2), padding='same')(xl4u_)
    xl3u_ = Concatenate()([xl3_, xl3u_])
    # xl3u_ = conv2D(b, xl3u_)
    xl3u_ = conv2D(c, xl3u_)
    xl3u = xl3u_    


    #xl3u_ #up
    #xl3u_ pass
  


    xl2u_ = tf.keras.layers.Conv2DTranspose(b, (2, 2), strides=(2, 2), padding='same')(xl3u_)
    xl2u_ = Concatenate()([xl2_, xl2u_])
    # xl2u_ = conv2D(a, xl2u_)
    xl2u_ = conv2D(b, xl2u_)
    xl2u = xl2u_    


    #xl2u_ #up
    #xl2u_ pass
  

    xl1u_ = tf.keras.layers.Conv2DTranspose(a, (2, 2), strides=(2, 2), padding='same')(xl2u_)
    xl1u_ = Concatenate()([xl1_, xl1u_])
    # xl1u_ = conv2D(f, xl1u_)
    xl1u_ = conv2D(a, xl1u_)
    xl1u = xl1u_   


    k = 3  
   
    iw4 = IDWT2D(wave=Ψ)(Concatenate()([xl4u,h4]))
    #
    iw3 = tf.keras.layers.Conv2D(1, (k, k), activation=activation3, padding='same')(Concatenate()([xl3u,iw4]))
    # iw3 = SpatialAttention()(Concatenate()([xl3u, l3, iw4]))
    iw3 = IDWT2D(wave=Ψ)(Concatenate()([iw3,h3]))
    #
    iw2 = tf.keras.layers.Conv2D(1, (k, k), activation=activation3, padding='same')(Concatenate()([xl2u,iw3]))
    # iw2 = SpatialAttention()(Concatenate()([xl2u, l2, iw3]))
    iw2 = IDWT2D(wave=Ψ)(Concatenate()([iw2,h2]))
    #
    iw1 = tf.keras.layers.Conv2D(1, (k, k), activation=activation3, padding='same')(Concatenate()([xl1u,iw2]))
    # iw1 = SpatialAttention()(Concatenate()([xl1u, l1, iw2]))
    R = IDWT2D(wave=Ψ)(Concatenate()([iw1,h1]))    
   
    
    
    ## OLD
    # R = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(R)  
    
    ## NEW
    # s = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(s)
    if residual==True:
        R = Concatenate()([s, R])
    
    R = tf.keras.layers.Conv2D(n_classes, (1, 1), activation="softmax")(R)

    outputs = R

     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    if compile==True:
        # model = tf.keras.Model(inputs=[inputs], outputs=[x])
        # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        # model.compile(optimizer=generator_optimizer, loss=generator_loss)
       
        model.compile(optimizer=optimizer, loss=loss)
        return model
        #
    else: 
        return model


if __name__=='__main__':
    # Loss = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5) ## BAD loss
    # generator_loss = 'binary_crossentropy'
    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    Gφψ(compile=False).summary()
    Gφψ(input_shape=(128,128,1), residual=True, compile=False).summary()
    # Gφψ(compile=False).summary()