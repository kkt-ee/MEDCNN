import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class BoundaryAwareDiceLoss(tf.keras.losses.Loss):
    """Boundary aware Dice Loss 
    
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
    def __init__(self, alpha=1, beta=1, gamma=1, epsilon=1e-5, **kwargs):
        super(BoundaryAwareDiceLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice_loss = 1 - (2 * intersection + self.beta) / (union + self.alpha + self.gamma + self.epsilon)
        return dice_loss

# Example usage
# loss_function = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5)

# Assuming y_true and y_pred are your ground truth and predicted masks, respectively
# loss_value = loss_function(y_true, y_pred)
# loss_function = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5)

def custom_BAD_loss(y_true, y_pred):
    BAD_loss = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5)
    return 0.01*tf.keras.metrics.binary_crossentropy(y_true, y_pred) + 0.99*BAD_loss(y_true, y_pred)