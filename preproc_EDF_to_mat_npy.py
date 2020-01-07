
# script to extract and compare eeg data between two projects (TRIO & Julie)
# 0 read edf files
# 1 extract sampling rate
# 2 extract number of channels
# save in data frame
import numpy as np
import scipy.io as sio
from mne.io import read_raw_edf
import os

#set path to your EDF files
path='/home/tarek/sleepclassification/data_G_test/'

#set path where you want to save your mat or npy files
save_path='/home/tarek/sleepclassification/data_G_test/'

#select either you want to save mat or npy format
output_file='npy'

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".edf"):
            #read edf files and extract informations
            raw=read_raw_edf(os.path.join(root,filename))
            raw.load_data()
            print ('loading :' ,filename)
            #select electrodes you need to use
            EEG_data=raw.pick_types(meg=False, eeg=True, stim=False, eog=False, ecg=False, emg=False)
            data,time=EEG_data[:] #extract time series of selected electrodes
            del raw
            #save time series into mat or npy file
            save_file=save_path+filename.replace('.edf','')
            if output_file=='mat':
                sio.savemat(save_file,{'Data':data})
            elif output_file=='npy':
                np.save(save_file,data)
