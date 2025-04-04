import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class BoundaryAwareDiceLoss(tf.keras.losses.Loss):
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