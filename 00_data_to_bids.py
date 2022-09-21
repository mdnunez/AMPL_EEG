# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:01:05 2022

@author: Bronz
"""

import sys
import os

import pandas as pd

from mne import find_events
from mne.io import read_raw_bdf
from mne.utils import logger
from mne_bids import BIDSPath, write_raw_bids

from config import (
    FPATH_DATA_SOURCEDATA,
    FPATH_DATA_BIDS,
    FPATH_SOURCEDATA_NOT_FOUND_MSG,
    FNAME_SOURCEDATA_TEMPLATE,
    SUBJECT_IDS,
    montage
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subj = 3
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

if not os.path.exists(FPATH_DATA_SOURCEDATA):
    raise RuntimeError(
        FPATH_SOURCEDATA_NOT_FOUND_MSG.format(FPATH_DATA_SOURCEDATA)
    )
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
# path to file in question (i.e., which subject and session)
fname = FNAME_SOURCEDATA_TEMPLATE.format(subj=subj,exptype="eyesclosed2", dtype='eeg', ext='.bdf')

# %%
# 1) import the data
raw = read_raw_bdf(fname, preload=False)
# get sampling frequency
sfreq = raw.info['sfreq']
# channels names
channels = raw.info['ch_names']
import mne
montage = mne.channels.make_standard_montage('biosemi64')

# identify channel types based on matching names in montage

#from skimage.util import montage
types = []
for channel in channels:
    if channel in montage.ch_names:
        types.append('eeg')
    elif channel.startswith('EOG') | channel.startswith('EXG'):
        types.append('eog')
    else:
        types.append('stim')

# add channel types and eeg-montage
raw.set_channel_types(
    {channel: typ for channel, typ in zip(channels, types)})
raw.set_montage(montage)

# %%
#2)subject info
#this part is for the subject information and it si not used for now as we don't have these information
"""
# compute approx. date of birth
# get measurement date from dataset info
date_of_record = raw.info['meas_date']
# convert to date format
date = date_of_record.strftime('%Y-%m-%d')

# here, we compute only and approximate of the subject's birthday
# this is to keep the date anonymous (at least to some degree)
demographics = FNAME_SOURCEDATA_TEMPLATE.format(subj=subj,
                                                dtype='demographics',
                                                ext='.tsv')
demo = pd.read_csv(demographics, sep='\t', header=0)
age = demo[demo.subject_id == 'sub-' + str(subj).rjust(3, '0')].age
sex = demo[demo.subject_id == 'sub-' + str(subj).rjust(3, '0')].sex

year_of_birth = int(date.split('-')[0]) - int(age)
approx_birthday = (year_of_birth,
                   int(date[5:].split('-')[0]),
                   int(date[5:].split('-')[1]))

# add modified subject info to dataset
raw.info['subject_info'] = dict(id=subj,
                                sex=int(sex),
                                birthday=approx_birthday)

# frequency of power line
raw.info['line_freq'] = 50.0
"""
# %%
# 3) get eeg events
events = find_events(raw,
                     stim_channel='Status',
                     output='onset',
                     min_duration=0.0, initial_event=True)
event_id={"eventid":69376}
# only keep relevant events
#as there are no events this part is not used and only the event id of the initial event is written in a dictionary
#keep_evs = [events[i, 2] in event_id.values() for i in range(events.shape[0])]
#events = events[keep_evs]

# %%
# 4) export to bids
# create bids path
output_path = BIDSPath(subject=f'{subj:04}',
                       task="RestEyesClosed2",
                       datatype='eeg',
                       root=FPATH_DATA_BIDS)

# write file
write_raw_bids(raw,
               events_data=events, event_id=event_id,
              bids_path=output_path,
               overwrite=overwrite)