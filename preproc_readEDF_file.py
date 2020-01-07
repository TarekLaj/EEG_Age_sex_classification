# script to extract and compare eeg data between two projects (TRIO & Julie)
# 0 read edf files
# 1 extract sampling rate
# 2 extract number of channels
# save in data frame
import numpy as np
import pandas as pd

from mne.io import read_raw_edf
import os

#path='/media/karim/ADATA HV620/Jessica_db/'
#path='/media/karim/ADATA HV620/TRIO_db/Controls/'
path='/home/karim/sleepclassification/'
subject='10005n0_caf200_Stage_dépistage-a.edf'
annot_file='/home/karim/sleepclassification/10005n0_caf200_Stage_dépistage-a.edf.xml'

coln=('Subject_id','Number of EEG channels','EEG channel names','Sampling frequency')
df=pd.DataFrame(columns=coln)
Fs=[]
channel_names=[]
nb_ch=[]
sub_name=[]
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".edf"):
#read edf files and extract informations
            raw=read_raw_edf(os.path.join(root,filename),preload=True)
            raw.load_data()
            print ('loading :' ,filename)
            EEG_data=raw.pick_types(eeg=True)
            Fs.append(raw.info.get('sfreq'))
            channel_names.append(EEG_data.ch_names)
            nb_ch.append(len(EEG_data.ch_names))
            sub_name.append(filename)
            #del raw


df['Subject_id']=sub_name
df['Number of EEG channels']=nb_ch
df['EEG channel names']=channel_names
df['Sampling frequency']=Fs
#df.to_excel(path+'EDF_recap.xlsx')
#import pyedflib as pyedf
#file = pyedf.EdfReader(os.path.join(root,filename))  
#annotations = file.readAnnotations()  
#print(annotations)  