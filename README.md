# AMPL_EEG
EMGfunctions.py python code analyzes left and right EMG data. This code corrects the mean value of the data, filters it, rectifies it and plots it each time. This data analysis is according to the time and time vector is created in the code. It is from the beginning until the end but it is possible to check a specific time with time filters.
There are 4 functions: remove_mean, emg_filter, emg_rectify, altogether. Altogether function rectifies and filters emg data and the data with corrected mean is used in this function. 
In this code the last part should be changed to analyze right or left EMG data. 


Paths.json file: This file is formed to create the paths of the files listed above. Thus, paths.json creates bidsdata, root, sourcedata, derivatives directories for the mne python code.

Config.py file: This is a python code written to determine the values and parameters in the main mne python code. This includes naming format of the file as well as defining and naming the paths. Paths.json is used. Config is imported later to the code.
Line 50 of config.py may need to be changed depending upon the number of participants.

00_data_to_bids.py file: This file takes the data from “sourcedata” file (the bdf eeg data is taken from this file and transformed into BIDS format on mne python) and converts it to a more organized BIDS format where it is possible to see electrode locations, channels and data. This file gets channel names, events and organizes them.

01_run_preprocessing.py: This file does data processing. It filters the data, removes bad channels providing a better data analysis. These processed data is saved into the file called “derivatives”.

mne iclabel code.py: this code tests iclabel but it gives ""NoneType" object has not attribute "get_positions"" error. Montage is checked, it work separately but not in this code. 


