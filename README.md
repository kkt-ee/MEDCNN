# MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network 

[![PyPI Version](https://img.shields.io/pypi/v/TFDWT?label=PyPI&color=gold)](https://pypi.org/project/MEDCNN/) 
[![PyPI Version](https://img.shields.io/pypi/pyversions/MEDCNN)](https://pypi.org/project/MEDCNN/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15--2.19-darkorange)](https://www.tensorflow.org/)
[![TFDWT Version](https://img.shields.io/badge/TFDWT-0.0.2-orange)](https://pypi.org/project/TFDWT/)
[![Keras Version](https://img.shields.io/badge/keras-2--3-darkred)](https://keras.io/)
[![CUDA Version](https://img.shields.io/badge/cuda-12.5.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![NumPy Version](https://img.shields.io/badge/numpy-2.0.2-blueviolet)](https://numpy.org/)
[![MIT](https://img.shields.io/badge/license-GPLv3-deepgreen.svg?style=flat)](https://github.com/kkt-ee/TFDWT/LICENSE)


  - This is a 2D version of MEDCNN and **without attentions** in the decoder
  - Full paper with attentions: https://doi.org/10.1109/ICASSP49660.2025.10890832



<br/><br/><br/>

## Installation guide

<br/>

**Install TFDWT from PyPI** (Option $1$)

```bash
pip install MEDCNN
```

  
<br/>

**Install TFDWT from Github** (Option $2$)

Download the package

```bash
git clone https://github.com/kkt-ee/MEDCNN.git
```

Change directory to the downloaded MEDCNN

```bash
cd MEDCNN
```

Run the following command to install the TFDWT package

```bash
pip install .
```



<br/><br/><br/>



## Verify installation 

```python
import MEDCNN
MEDCNN.__version__
```

<br/><br/><br/>

## Sample usage

   - Import MEDCNN 2D Gφψ without attention
   ```python
   from MEDCNN.models.G2DwithoutAttention import Gφψ, configs
   ```

   - Import the control Unet2D model for reference
   ```python
   from MEDCNN.models.ControlUnet2D import Unet2D, uconfigs
   ```

   - Import utils to compile and train model
   ```python
   from MEDCNN.utils.utils import elapsedtime, timestamp
   from MEDCNN.utils.BoundaryAwareDiceLoss import BoundaryAwareDiceLoss
   from MEDCNN.utils.Load2Ddata import load_ibsr_XY
   from MEDCNN.utils.TTViterators import get_train_test_val_iterators
   from MEDCNN.utils.dice import dice_coef
   from MEDCNN.utils.compile1 import compile_model
   from MEDCNN.utils.Train1 import train
   ```

   - Example: Compile a MEDCNN
   ```python
   CONFIGKEY= 'minimal2'
   model, segconfig = Gφψ(config=configs[CONFIGKEY], compile=False), 'nonResidual'
   model, lossname = compile_model(model, dataset, dice_coef)
   ```

   - Example: Compile a control Unet2D
   ```python
   CONFIGKEY = '45678',
   model, segconfig = Unet2D(config=uconfigs['45678'], compile=False), 'nonResidual'
   model, lossname = compile_model(model, dataset, dice_coef)
   ```

   - Example: Train a model with X an Y of shape (7056, 256, 256, 1), (7056, 256, 256, 1)
   ```python
   train_iterator, test_iterator, val_iterator = get_train_test_val_iterators(X,Y) #Assuming X and Y is loaded by a dataloader
   train(
    model, 
    train_iterator, test_iterator, val_iterator, 
    dataset='IBSR', 
    segconfig=segconfig, 
    lossname='bce', 
    CONFIGKEY=CONFIGKEY, 
    epochs=40)
   ```





<br/><br/><br/>

* * *

## Uninstall MEDCNN

```bash
pip uninstall MEDCNN
```

  
<br/><br/><br/><br/><br/>

* * *

***MEDCNN (C) 2025 Kishore Kumar Tarafdar, Prime Minister's Research Fellow, EE, IIT Bombay, भारत*** 🇮🇳