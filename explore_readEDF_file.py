# script to extract and compare eeg data between two projects (TRIO & Julie)
# 0 read edf files
# 1 extract sampling rate
# 2 extract number of channels
# save in data frame
import numpy as np
import pandas as pd

from mne.io import read_raw_edf
import os
usr='tarek'
#path_to_dd=os.path.join('/media',usr,'ADATA HV620')
#if not os.path.ismount(path_to_dd) :
#    raise Exception('Please Plug ADATA HV620 hard drive !')
    
#database='TRIO_db'
#classe='Controls'
#path=os.path.join(path_to_dd,database,classe)

path='/home/tarek/sleepclassification/data_G_test/'
coln=('Subject_id','Number of EEG channels','EEG channel names','Sampling frequency')
df=pd.DataFrame(columns=coln)
Fs=[]
channel_names=[]
nb_ch=[]
sub_name=[]
print(path)
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".edf"):
#read edf files and extract informations
            raw=read_raw_edf(os.path.join(root,filename))
            raw.load_data()
            print ('loading :' ,filename)
            EEG_data=raw.pick_types(meg=False, eeg=True, stim=False, eog=False, ecg=False, emg=False)
            Fs.append(raw.info.get('sfreq'))
            channel_names.append(EEG_data.ch_names)
            nb_ch.append(len(EEG_data.ch_names))
            sub_name.append(filename)
            #del raw


df['Subject_id']=sub_name
df['Number of EEG channels']=nb_ch
df['EEG channel names']=channel_names
df['Sampling frequency']=Fs
#save_df='/home/tarek/sleepclassification/recap_edf_{dd}_{cl}.csv'.format(dd=database,cl=classe)
#df.to_csv(save_df, encoding='utf8')
