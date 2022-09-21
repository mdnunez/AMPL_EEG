# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:59:12 2022

@author: Bronz
"""



from mne.io import read_raw_bdf
raw = read_raw_bdf('S0002_eyesclosed_20220804_01.bdf', preload=True)

#raw.plot_psd(fmax=50)


"""
from mne.viz import plot_raw
graph= plot_raw
fig = raw.plot() 
"""
import numpy as np
raw.notch_filter(50*np.arange(1,11))
raw.filter(1,50.)
filt_raw=raw.copy()


import mne
montage = mne.channels.make_standard_montage('biosemi64')
filt_raw.set_montage(montage)
montage.plot()


import mne
ica=mne.preprocessing.ICA (n_components=20)
x=ica.fit(filt_raw)
ica.plot_components(outlines="skirt")
ica.plot_properties(filt_raw)

"""
from mne_icalabel import label_components

# assuming you have a Raw and ICA instance previously fitted
label_components(raw, ica, method='iclabel')


ica.plot_overlay(raw, exclude=[0], picks="eeg")
"""

import importlib

importlib.import_module('mne_icalabel')


from mne_icalabel import label_components
ic_labels = label_components(raw, ica, method="iclabel")
print(ic_labels)








