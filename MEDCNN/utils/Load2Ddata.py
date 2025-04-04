# IBSR_allplanes_shuffled_resized_INPUTS_SS.npy 
# IBSR_allplanes_shuffled_resized_TARGETS_SS.npy
import os
import numpy as np

def load_ibsr_XY(datapath):
    """Load IBSRv2 data
    
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
    dataset = 'IBSR'
    # datapath = '/home/k/LOTUS/DATAMRI/IBSR_cooked_XY_SkullStripping'
    # datapath = '/home/kishor/FILESHARE/IBSR_cookedXY_allplanes/IBSR_XY_skullstripping/'
    # datapath = '/data1/kishoretarafdar/FILESHARE/IBSR_cookedXY_allplanes/IBSR_XY_skullstripping'
    # with open('IBSR_allplanes_shuffled_resized_INPUTS_SS.npy', 'rb') as f:
    with open(os.path.join(datapath, 'IBSR_X.npy'), 'rb') as f:
        X = np.load(f)
    del f
    # with open('IBSR_allplanes_shuffled_resized_TARGETS_SS.npy', 'rb') as f:
    with open(os.path.join(datapath, 'IBSR_Y.npy'), 'rb') as f:
        Y = np.load(f)
    del f#, datapath
    # X.shape, Y.shape
    return X, Y, dataset


def load_atlas_XY(datapath):
    """Load ATLAS2 data 
    
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
    dataset = 'ATLAS'
    with open(os.path.join(datapath, 'ATLAS_X_minnorm40.npy'), 'rb') as f:
        X = np.load(f)
    del f 
    with open(os.path.join(datapath, 'ATLAS_Y_minnorm40_int8.npy'), 'rb') as f:
        Y = np.load(f)
    del f
    # X.shape, Y.shape 
    return X, Y, dataset