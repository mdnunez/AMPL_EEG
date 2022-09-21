# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:12:31 2022

@author: Bronz
"""

import os

from pathlib import Path

import numpy as np

import json

from mne.channels import make_standard_montage

# get path to current file
parent = Path(__file__).parent.resolve()

# -----------------------------------------------------------------------------
# file paths
with open(os.path.join(parent, 'paths.json')) as paths:
    paths = json.load(paths)

# the root path of the dataset
FPATH_DATA = paths['root']
# path to sourcedata (biosemi files)
FPATH_DATA_SOURCEDATA = Path(paths['sourcedata'])
# path to BIDS compliant directory structure
FPATH_DATA_BIDS = Path(paths['bidsdata'])
# path to derivatives
FPATH_DATA_DERIVATIVES = Path(paths['derivatives'])

# -----------------------------------------------------------------------------
# file templates
# the path to the sourcedata directory
FNAME_SOURCEDATA_TEMPLATE = os.path.join(
    str(FPATH_DATA_SOURCEDATA),
    "sub-{subj:04}",
    "{dtype}",
    "sub-{subj:04}_{exptype}_{dtype}{ext}"
)

# -----------------------------------------------------------------------------
# problematic subjects
NO_DATA_SUBJECTS = {}

# originally, subjects from 1 to 151, but some subjects should be excluded
SUBJECT_IDS = np.array(list(set(np.arange(1, 53)) - set(NO_DATA_SUBJECTS)))

# -----------------------------------------------------------------------------
# default messages
FPATH_SOURCEDATA_NOT_FOUND_MSG = (
    "Did not find the path:\n\n>>> {}\n"
    "\n>>Did you define the path to the data on your system in `config.py`? "
    "See the FPATH_DATA_SOURCEDATA variable!<<\n"
)

FPATH_BIDS_NOT_FOUND_MSG = (
    "Did not find the path:\n\n>>> {}\n"
    "\n>>Did you define the path to the data on your system in `config.py`? "
    "See the FPATH_DATA_BIDS variable!<<\n"
)

FPATH_DERIVATIVES_NOT_FOUND_MSG = (
    "Did not find the path:\n\n>>> {}\n"
    "\n>>Did you define the path to the data on your system in `config.py`? "
    "See the FPATH_DATA_DERIVATIVES variable!<<\n"
)

EOG_COMPONENTS_NOT_FOUND_MSG = (
    "No {type} ICA components found for subject {subj}"
)
#%%
montage = make_standard_montage(kind='biosemi64')
