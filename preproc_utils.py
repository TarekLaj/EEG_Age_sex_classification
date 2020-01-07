from xml.etree import ElementTree
import pandas as pd
from mne.io import read_raw_edf
import numpy as np
import scipy.io as sio


def extract_data_fromEDF(EDF_file):
    raw=read_raw_edf(EDF_file)
    raw.load_data()
    data_pick=raw.pick_types(eeg=True,eog=True,emg=True)
    data,time=data_pick[:]
    channel_names=raw.ch_names
    Fs=raw.info.get('sfreq')
    return data,channel_names,Fs
def select_data(data,channels_names,data_to_select):
    print('selecting {} data'.format(data_to_select))
    EEG_channel_names,EEG_index=[],[]
    EOG_channel_names,EOG_index=[],[]
    ECG_channel_names,ECG_index=[],[]
    EMG_channel_names,EMG_index=[],[]
    for ch in channels_names:
        if 'EEG' in ch:
            if 'LOC' not in ch and 'ROC' not in ch:
                EEG_channel_names.append(ch)
                EEG_index.append(channels_names.index(ch))
        elif 'EOG' in ch:
                EOG_channel_names.append(ch)
                EOG_index.append(channels_names.index(ch))
        elif 'ECG' in ch:
                ECG_channel_names.append(ch)
                ECG_index.append(channels_names.index(ch))
        elif 'EMG' in ch:
                EMG_channel_names.append(ch)
                EMG_index.append(channels_names.index(ch))
    if data_to_select=='EEG':

        selected_data=data[EEG_index,:]
        selected_channel_names=EEG_channel_names
    elif data_to_select=='EOG':
        selected_data=data[EOG_index,:]
        selected_channel_names=EOG_channel_names
    elif data_to_select=='ECG':
        selected_data=data[ECG_index,:]
        selected_channel_names=ECG_channel_names
    elif data_to_select=='EMG':
        selected_data=data[EMG_index,:]
        selected_channel_names=EMG_channel_names
    return selected_data,selected_channel_names
def extract_markers(xml_file,save,save_path,subject_name):
    print('Extracting hypnogramme of subject {}'.format(subject_name))
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    SleepStages=[]
    for scores in root.findall('SleepStages'):
        for sc in scores:
            SleepStages.append(sc.text)


    Art_ch,Art_start,Art_duration=[],[],[]
    for scores in root.iter('ScoredEvent'):
       names_temp=scores.find('Name').text
       if 'HarmAct_' in names_temp:
           Art_ch.append(names_temp)
           Art_start.append(scores.find('Start').text)
           Art_duration.append(scores.find('Duration').text)

    if save==True:
        xls_file=save_path+'Hyp_{}.xlsx'.format(subject_name)
        dt={'Scoring': SleepStages}
        df=pd.DataFrame(data=dt)
        df.to_excel(xls_file)
        xls_file1=save_path+'Artf_{}.xlsx'.format(subject_name)
        dt1={'Artefact_channel': Art_ch,'Artefact_start':Art_start,
             'Artefact_duration':Art_duration}
        df1=pd.DataFrame(data=dt1)
        df1.to_excel(xls_file1)




    Art_duration=np.asarray(list(map(float, Art_duration)))
    Art_start=np.asarray(list(map(float, Art_start)))
    Art_start=np.reshape(Art_start,[len(Art_start),1])
    SleepStages = np.asarray(list(map(int, SleepStages)))
    return SleepStages,Art_ch,Art_start,Art_duration
def data_epoching(data=[],epoch_length=30,Fs=128):
    print('Epoching data into {} sec segments...' .format(str(epoch_length)) )
    if data.shape[0]>data.shape[1]:
        print('warning data shape must be elect*time...')
        print('reshaping data')
        data=data.T
    n=epoch_length*Fs
    L=data.shape[1]
    a_extrat=int((-L)%n)
    x_temp=np.concatenate((data,np.zeros((data.shape[0],a_extrat))),axis=1)
    nbre_epochs=int(x_temp.shape[1]/n)
    X=np.split(x_temp,nbre_epochs,axis=1)
    return np.dstack(X)
#    if data.shape[0]>data.shape[1]:
#        data=data.T
#
#    n=int(Fs)*epoch_length
#    m=0
#
#    Segments=np.array([])
#    while n<=data.shape[1]:
#        #D=data[:,m:n]
#        Segments=np.dstack((Segments,data[:,m:n])) if Segments.size else data[:,m:n]
#
#        n=n+epoch_length*int(Fs);
#        m= m+epoch_length*int(Fs);
#    return Segments


def extract_data_per_sleep_stage(Segments,hypnograme,ref):
    print('Extracting data per sleep stage using {} manual'.format(ref))
    if ref=='AASM':
        AWAKE=Segments[:,:,hypnograme==0]
        N1=Segments[:,:,hypnograme==1]
        N2=Segments[:,:,hypnograme==2]
        N3=Segments[:,:,hypnograme==3]
        REM=Segments[:,:,hypnograme==5]
        return AWAKE, N1, N2,N3,REM
    elif ref=='K&R':
        AWAKE=Segments[:,:,hypnograme==0]
        S1=Segments[:,:,hypnograme==1]
        S2=Segments[:,:,hypnograme==2]
        S3=Segments[:,:,hypnograme==3]
        S4=Segments[:,:,hypnograme==4]
        REM=Segments[:,:,hypnograme==5]
        return S1,S2,S3,S4,REM


def remove_artefacts(data=[],hyp=[],Art_start=None,epoch_length=30):
    print('Removing artefacted EEG epochs....')
    artf_ep=np.unique( (Art_start/epoch_length).astype(int))

    clean_data=np.delete(data,artf_ep,axis=2)
    clean_hyp=np.delete(hyp,artf_ep)

    return clean_data,clean_hyp



def save_data(data_to_save,save_path,save_format):
    print ('Saving data in {} format'.format(save_format))
    if save_format=='mat':
        sio.savemat(save_path,{'Data':data_to_save})
    elif save_format=='npy':
        np.save(save_path,data_to_save)
