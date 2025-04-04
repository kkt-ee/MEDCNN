#Build the model
import tensorflow as tf

uconfigs = {
    '12345':(2,4,8,16,32),
    '23456':(4,8,16,32,64),
    '34567':(8,16,32,64,128),
    '45678':(16,32,64,128,256),
    '56789':(32,64,128,256,512)
    }    
    
    #a,b,c,d,e=2,4,8,16,32
    # a,b,c,d,e=4,8,16,32,64
    # a,b,c,d,e=8,16,32,64,128
    # a,b,c,d,e=16,32,64,128,256

def Unet2D(
    n_classes=2, 
    n_input_channels=1, 
    scale=1, 
    input_shape =(256, 256, 1),
    # dataset='IBSR', 
    # config=configs['12345'],
    config=uconfigs['45678'],
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    residual=False,
    compile=True):

    a, b, c, d, e = config
    
    #inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # inputs = tf.keras.layers.Input((256, 256, 1))
    # s=inputs
    inputs = tf.keras.layers.Input(input_shape)
    # s = tf.keras.layers.Resizing(
    #                 height=256,
    #                 width=256,
    #                 interpolation='bilinear',
    #                 crop_to_aspect_ratio=False
    #             )(inputs)


    s = inputs
    s = tf.keras.layers.Lambda(lambda x: x / 255)(s)
    # s = tf.keras.layers.Lambda(lambda x: tf.where(
    #     tf.reduce_sum(x) !=0,
    #     (((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)) - (tf.reduce_min((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)))) 
    #    / ((tf.reduce_max((x - tf.reduce_mean(x)) / tf.math.reduce_std(x))) - (tf.reduce_min((x - tf.reduce_mean(x)) / tf.math.reduce_std(x)))), 
    #     x
    #     ))(s)
    #s = Normlayer(inputs_resiz)

    #resize
    #inputs_resized = tf.keras.layers.Resizing(
    #    height=256,
    #    width=256,
    #    interpolation='bilinear',
    #    crop_to_aspect_ratio=False
    #)(inputs)


    #s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #a,b,c,d,e=2,4,8,16,32
    # a,b,c,d,e=4,8,16,32,64 ##it was here
    # a,b,c,d,e=8,16,32,64,128
    # a,b,c,d,e=16,32,64,128,256
    #Contraction path
    c1 = tf.keras.layers.Conv2D(a, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(a, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(c, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(c, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(d, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(d, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    #bottom
    c5 = tf.keras.layers.Conv2D(e, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(e, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(d, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(d, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(d, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(c, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(c, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(c, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(b, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(a, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(a, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(a, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # out_resiz  = tf.keras.layers.Resizing(
    #                 height=256,
    #                 width=192,
    #                 interpolation='bilinear',
    #                 crop_to_aspect_ratio=False
    #             )(c9)

    # out_resiz=c9
    # outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(out_resiz)

    R = c9
    if residual==True:
        R = Concatenate()([s, R])
    R = tf.keras.layers.Conv2D(n_classes, (1, 1), activation="softmax")(R)
    outputs = R

     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])





    
    # model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    #model.compile(optimizer='adam', loss=IoU, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    # model.compile(optimizer='adam', loss=Dice, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

    # if compile==True:
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #     # iou_metric = tf.keras.metrics.MeanIoU(num_classes=2, name='binaryIoU') 
    #     iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='IoU')
    #     model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy', dice_coef, iou_metric])

    if compile==True:
        # model = tf.keras.Model(inputs=[inputs], outputs=[x])
        # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        # model.compile(optimizer=generator_optimizer, loss=generator_loss)
       
        model.compile(optimizer=optimizer, loss=loss)
        return model
        #
    else: 
        return model





    # model.summary()
    # plot_model(model, 
    #         show_shapes=True,
    #         expand_nested=False,
    #         show_layer_activations=True)
            #    to_file=f'Unet_{a}-{b}-{d}-{d}-{e}-{d}-{c}-{b}-{a}{round(time.time(),4)}.png')



if __name__=='__main__':
    # Loss = BoundaryAwareDiceLoss(alpha=1, beta=1, gamma=1, epsilon=1e-5) ## BAD loss
    # generator_loss = 'binary_crossentropy'
    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    Unet2D(compile=False).summary()
    Unet2D(input_shape=(128,128,1), config=uconfigs['45678'], residual=True, compile=False).summary()
    # Gφψ(compile=False).summary()