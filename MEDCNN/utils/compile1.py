# # # include ../dirx 
# mylibpath = [
#       '/home/kishoretarafdar/src/MEDCNN_copy'
#     ]
# import sys
# [sys.path.insert(0,_) for _ in mylibpath]
# del mylibpath

import tensorflow as tf
# from BoundaryAwareDiceLoss import BoundaryAwareDiceLoss
# from dice import dice_coef
from keras.utils import plot_model 

def compile_model(model, dataset, dice_coef):
    """Compile model 
    
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
    if dataset == 'IBSR':
        # Loss, lossname = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5), 'bad' ## BAD loss
        Loss, lossname = 'binary_crossentropy', 'bce'
    elif dataset == 'ATLAS':
        Loss, lossname = 'binary_crossentropy', 'bce'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=(0,1), name='IoU')

    model.compile(optimizer=optimizer, loss=Loss, metrics=['accuracy', dice_coef, iou_metric], jit_compile=False)
    model.summary()
    plot_model(model, 
        show_shapes=True, 
        expand_nested=False, 
        show_layer_activations=True)
        # #   to_file=f'{wave}_Unet_{round(time.time(),4)}.png')
    return model, lossname