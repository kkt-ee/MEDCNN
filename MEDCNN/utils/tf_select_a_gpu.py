import tensorflow as tf

def select_a_gpu(gpus:list, gpu_id:int, memory_limit=47):
    """Hard limit: 47 GB max gpu memory

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
    _select_gpu = gpus[gpu_id]
    if gpus:
        try:
            """limit memory to the seleced GPU"""
            # _limitGB = 32
            memory_limit = int(memory_limit*1024)
            tf.config.set_logical_device_configuration(
                _select_gpu, 
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])

            """Restrict tensorflow to use 1 GPU"""
            tf.config.set_visible_devices(_select_gpu, 'GPU')
            one_logical_gpu = tf.config.list_logical_devices('GPU')


            print(f"{len(gpus)} Physical GPUs available \nSelected {len(one_logical_gpu)} Logical GPU with {int(memory_limit/1024)} GB memory limit")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)



if __name__=='__main__':
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))  
    select_a_gpu(gpus, gpu_id=1, memory_limit=1)
    del select_a_gpu, gpus



"""Other examples: Cluster of virtual gpus in a GPU chip

    gpus = tf.config.list_physical_devices('GPU')
    _select_gpu = gpus[gpu_id]
    tf.config.set_logical_device_configuration(
        _select_gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
            tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
"""