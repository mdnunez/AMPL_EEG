# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:37:27 2022

@author: Bronz
"""

import sys
import os

from pathlib import Path

import json

import numpy as np
import matplotlib.pyplot as plt

from mne import events_from_annotations, concatenate_raws
from mne.preprocessing import ICA, corrmap
from mne.utils import logger

from mne_bids import BIDSPath, read_raw_bids

from config import (
    FPATH_DATA_BIDS,
    FPATH_DATA_DERIVATIVES,
    FPATH_BIDS_NOT_FOUND_MSG,
    EOG_COMPONENTS_NOT_FOUND_MSG,
    SUBJECT_IDS
   
)

from utils import parse_overwrite

from pyprep.prep_pipeline import PrepPipeline

# %%
# default settings (use subject 1, don't overwrite output files)
subj = 1
overwrite = False

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        sub=subj,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    subj = defaults["sub"]
    overwrite = defaults["overwrite"]

# %%
# paths and overwrite settings
if subj not in SUBJECT_IDS:
    raise ValueError(f"'{subj}' is not a valid subject ID.\nUse: {SUBJECT_IDS}")

if not os.path.exists(FPATH_DATA_BIDS):
    raise RuntimeError(
        FPATH_BIDS_NOT_FOUND_MSG.format(FPATH_DATA_BIDS)
    )
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
# create bids path for import
str_subj = str(subj).rjust(4, '0')
raw_fname = BIDSPath(root=FPATH_DATA_BIDS,
                     subject=str_subj,
                     task='RestNoEyes1',
                     datatype='eeg',
                     extension='.bdf')
# get the data
raw = read_raw_bids(raw_fname)
raw.load_data()

# get sampling rate
sfreq = raw.info['sfreq']

# get montage
montage = raw.get_montage()

# %%
# extract relevant parts of the recording
# there aren't any events so this part of the code is not used until the task is determined
"""
# extract events
events = events_from_annotations(raw, event_id=task_events)

# extract cue events
cue_evs = events[0]
cue_evs = cue_evs[(cue_evs[:, 2] >= 1) & (cue_evs[:, 2] <= 7)]

# latencies and difference between two consecutive cues
latencies = cue_evs[:, 0] / sfreq
diffs = [(y - x) for x, y in zip(latencies, latencies[1:])]

# get first event after a long break (i.e., when the time difference between
# stimuli is greater than 10 seconds). This should only be the case in between
# task blocks
breaks = [diff for diff in range(len(diffs)) if diffs[diff] > 10]
logger.info("\nIdentified breaks at positions:\n %s " % ', '.join(
    [str(br) for br in breaks]))

# save start and end points of task blocks
# subject '041' has more practice trials (two rounds)
if subj == 41:
    # start of first block
    b1s = latencies[breaks[2] + 1] - 2
    # end of first block
    b1e = latencies[breaks[3]] + 6

    # start of second block
    b2s = latencies[breaks[3] + 1] - 2
    # end of second block
    b2e = latencies[breaks[4]] + 6

# all other subjects have the same structure
else:
    # start of first block
    b1s = latencies[breaks[0] + 1] - 2
    # end of first block
    b1e = latencies[breaks[1]] + 6

    # start of second block
    b2s = latencies[breaks[1] + 1] - 2
    # end of second block
    if len(breaks) > 2:
        b2e = latencies[breaks[2]] + 6
    else:
        b2e = latencies[-1] + 6
"""
# %%
#there aren't any events so this part of the code is not used until the task is determined
"""
# extract data chunks belonging to the task blocks and concatenate them
# block 1
raw_bl1 = raw.copy().crop(tmin=b1s, tmax=b1e)
# block 2
raw_bl2 = raw.copy().crop(tmin=b2s, tmax=b2e)
# concatenate
raw_bl = concatenate_raws([raw_bl1, raw_bl2])
del raw

# %%
# apply filter to data
raw_bl = raw_bl.filter(l_freq=0.1, h_freq=40.,
                       picks=['eeg', 'eog'],
                       filter_length='auto',
                       l_trans_bandwidth='auto',
                       h_trans_bandwidth='auto',
                       method='fir',
                       phase='zero',
                       fir_window='hamming',
                       fir_design='firwin',
                       n_jobs=4)
"""
# %%
# raw_bl.plot(scalings=dict(eeg=50e-6), n_channels=64, block=True)

# %%
# make a copy of the data in question
#raw_copy = raw_bl.copy()

# set up prep pipeline

prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(50, raw.info['sfreq'] / 2, 50),
    
}
# run data through preprocessing pipeline
prep = PrepPipeline(raw, prep_params, montage, ransac=False)
prep.fit()

# %%
# crate summary for PyPrep output
bad_channels = {'interpolated_chans': prep.interpolated_channels,
                'still_noisy': prep.still_noisy_channels,
                'ransac': prep.ransac_settings}


# %%
# export summary to .json

# create path
FPATH_BADS = os.path.join(FPATH_DATA_DERIVATIVES,
                          'preprocessing',
                          'bad_channels',
                          'sub-%s' % str_subj,
                          '%s_bad_channels_RestNoEyes1.json' % str_subj)
# chekc if directory exists
if not Path(FPATH_BADS).exists():
    Path(FPATH_BADS).parent.mkdir(parents=True, exist_ok=True)
# save file
with open(FPATH_BADS, 'w') as bads_file:
    json.dump(bad_channels, bads_file, indent=2)

# %%
# extract the re-referenced eeg data
clean_raw = prep.raw.copy()
del raw

# %%
# interpolate any remaining bad channels
clean_raw.interpolate_bads()
# apply notch filter (50Hz)
line_noise = [50., 100.]
clean_raw = clean_raw.notch_filter(freqs=line_noise, n_jobs=4)

# %%
# prepare ICA

# filter data to remove drifts
raw_filt = clean_raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=4)

# set ICA parameters
method = 'infomax'
reject = dict(eeg=250e-6)
ica = ICA(n_components=64,
          method=method,
          fit_params=dict(extended=True))

# run ICA
ica.fit(raw_filt,
        reject=reject,
        reject_by_annotation=True)

# %%
# look for components that show high correlation with the artefact templates
# see if the ica component belongs to the eye according to the ica template file with eye coordinations
"""
try:
    # lower the correlation threshold for subject 14
    # (allows corrmap to select 2 components for vertical eye movements)
    if subj == 14:
        threshold = 0.85
    else:
        threshold = 'auto'
    corrmap([ica],
            template=np.array(ica_templates['vertical_eye']),
            threshold=threshold, label='vertical_eog', show=False)
    plt.close('all')
except:
    logger.info(
        EOG_COMPONENTS_NOT_FOUND_MSG.format(
            type='vertical eye movement',
            subj=subj)
    )
finally:
    logger.info("\nDone looking for vertical eye movement components\n")

try:
    # raise the correlation threshold for subject 14
    # (makes corrmap very strict about potential horizontal eye movements
    # components)
    if subj == 14:
        threshold = 0.90
    else:
        threshold = 'auto'
    corrmap([ica],
            template=np.array(ica_templates['horizontal_eye']),
            label='horizontal_eog', show=False, threshold=threshold)
    plt.close('all')
except:
    logger.info(
        EOG_COMPONENTS_NOT_FOUND_MSG.format(
            type='horizontal eye movement',
            subj=subj)
    )
finally:
    logger.info("\nDone looking for horizontal eye movement components\n")
"""
# %%
"""
# get the identified components
bad_components = []
for label in ica.labels_:
    if subj != 14:
        # only take the first component that was identified by the template
        bad_components.extend([ica.labels_[label][0]])
    else:
        # only take the first component that was identified by the template
        bad_components.extend(ica.labels_[label])
logger.info('\n Found bad components:\n %s' % bad_components)

# add bad components to exclusion list
ica.exclude = np.unique(bad_components)
"""

# %%
# save ica figure

# create path
FPATH_ICA = os.path.join(FPATH_DATA_DERIVATIVES,
                          'preprocessing',
                          'ICA',
                          'sub-%s' % str_subj,
                          '%s_ica_components_RestNoEyes1.png' % str_subj)
# chekc if directory exists
if not Path(FPATH_ICA).exists():
    Path(FPATH_ICA).parent.mkdir(parents=True, exist_ok=True)

# save figure
fig = ica.plot_components(show=False)
fig[0].savefig(FPATH_ICA, dpi=100, facecolor='white')
plt.close('all')

# %%
# save emg data

# create path
FPATH_EMG = os.path.join(FPATH_DATA_DERIVATIVES,
                          'preprocessing',
                          'emg',
                          'sub-%s' % str_subj,
                          '%s_ica_components_RestNoEyes1.png' % str_subj)
# chekc if directory exists
if not Path(FPATH_EMG).exists():
    Path(FPATH_EMG).parent.mkdir(parents=True, exist_ok=True)
    
# save file
with open(FPATH_EMG, 'w') as emg_file:
    json.dump(bad_channels, bads_file, indent=2)
# %%
# remove the identified components
ica.apply(clean_raw)

# create path for preprocessed dara
FPATH_PREPROCESSED = os.path.join(FPATH_DATA_DERIVATIVES,
                                  'preprocessing',
                                  'preprocessed',
                                  'sub-%s' % str_subj,
                                  'sub-%s_preprocessed_RestNoEyes-raw.fif' % str_subj)
# chekc if directory exists
if not Path(FPATH_PREPROCESSED).exists():
    Path(FPATH_PREPROCESSED).parent.mkdir(parents=True, exist_ok=True)

# save file
clean_raw.save(FPATH_PREPROCESSED, overwrite=True)
