# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function


import os
import numpy as np
import tqdm
from .data_load import load_data
from .utils import load_spectrograms

# Load data
fpaths, _, _ = load_data()  # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists("mels"):
        os.mkdir("mels")
    if not os.path.exists("mags"):
        os.mkdir("mags")

    np.save("mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("mags/{}".format(fname.replace("wav", "npy")), mag)
