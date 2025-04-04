# datasetname = 'IBSR'
# datasetname = 'ATALS'

# # include ../dirx 
mylibpath = [
    # '/home/kishor/src/FastDWTConvLayers',
    # '/home/kishor/src/MRSegmentation/Attentions19102023'
      # '/home/k/src/_MEDCNNsrc_part/MEDCNN_copy'
      '/home/kishoretarafdar/src/MEDCNN_copy'
      
    #'/home/k/PLAYGROUND10GB/SKULSTRIPpaper__'
    ]
import sys
[sys.path.insert(0,_) for _ in mylibpath]
del mylibpath

from utils import elapsedtime, timestamp
import shelve
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def train(model, train_iterator, test_iterator, val_iterator, dataset='IBSR', segconfig='nonResidual' , lossname='bce', CONFIGKEY='4567', epochs=40):
    """Train model

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
    start_time = time.time()
    # MODEL_FNAME_PATTERN = f'{timestamp(start_time)}Ghaar_{datasetname}_{segconfig}_{lossname}_config{configkey}.keras'
    MODEL_FNAME_PATTERN = f'{timestamp(start_time)}G_{dataset}_{segconfig}_{lossname}_config{CONFIGKEY}.keras'
    # MODEL_FNAME_PATTERN = 'abc.keras'
    # MODEL_FNAME_PATTERN = f'{timestamp(start_time)}Grbio13_{datasetname}_{segconfig}_{lossname}_config{configkey}.keras'
    # model = dwtunet()
    # epochs =  40 # tot  = 120
    patience = epochs//2
    ###############################
    #Modelcheckpoint
    # checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_sagittal.h5', verbose=1, save_best_only=True)


    # patience = 10
    # patience = epochs #!!override early stopping
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FNAME_PATTERN, verbose=1, save_best_only=True),
    ]
            # tf.keras.callbacks.TensorBoard(log_dir='dwtlogs')]

    #results = model.fit(X_train_resized, Y_train_resized, validation_split=0.1, batch_size=100, epochs=25, callbacks=callbacks)
    #results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=20, epochs=1, callbacks=callbacks)
    results = model.fit(train_iterator, 
                        validation_data = val_iterator,
                        epochs=epochs, 
                        callbacks=callbacks)


    ####################################
    #TEST
    test_results = model.evaluate(test_iterator)#, batch_size=20)
    print("\n TEST:\ntest loss, test acc:", test_results)

    ## Show model weights distribution
    allp = [tf.reshape(_, [-1]).numpy() for _ in model.weights]
    len(allp)
    plt.figure(figsize=(60,8))
    plt.boxplot(allp)
    plt.show()
    del allp


    ## Time elapsed
    # end_time = time.time()
    elapsedtime(start_time, end_time=time.time())


    with shelve.open(f'G{CONFIGKEY}') as f:
        # Access variables stored in the shelve file
        f['train_result'] = results.history
        f['test_result'] = test_results

    
    """Saving results as pickles"""
    with open(f"train_result.pkl", "wb") as f:
        pickle.dump(results.history, f)
    with open(f"test_result.pkl", "wb") as f:
        pickle.dump(test_results, f)
        
    # del dataset, f, MODEL_FNAME_PATTERN, epochs, patience, start_time
    return results.history, test_results