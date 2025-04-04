import tensorflow as tf

def select_a_gpu(gpus:list, gpu_id:int, memory_limit=47):
    """Hard limit: 47 GB max gpu memory"""
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