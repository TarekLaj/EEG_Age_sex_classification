
from SleepCodes import select_data,save_data,extract_data_fromEDF,remove_artefacts
from SleepCodes import extract_markers,data_epoching,extract_data_per_sleep_stage
import os
from FeaturesExtraction import computePowerbandsEpochs
import numpy as np
#path=os.path.join('/home/karim','eeg_agegenderclassification')
path='/home/karim/eeg_agegenderclassification/'
#------------------------------- Parameters: ----------------------------------#

#subject_name='10005n0_caf200_Stage_dépistage-a'
subject_name='16008n1_Stage_Montage1'
xmlfile=path+subject_name+'.edf.xml'
edf_file=os.path.join(path,subject_name+'.edf')
epoch_length=20

#--------------------------load data and Extracting markers ------------------#
data,channel_names,Fs=extract_data_fromEDF(edf_file)
hyp,Art_ch,Art_start,Art_duration=extract_markers(xmlfile,True,path,subject_name)

#---------------------Select EEG data and dividing it into 30sec epochs-------#
EEG_data,ch_names=select_data(data,channel_names,'EEG')
Segments=data_epoching(data=EEG_data,epoch_length=epoch_length,Fs=Fs)
nbre_epoch=Segments.shape[2]
#--------------------------Remove Artefacts-----------------------------------#
clean_data,clean_hyp=remove_artefacts(data=Segments,hyp=hyp,Art_start=Art_start,epoch_length=epoch_length)
#--------------------------Computing spectral Power density-------------------#
#Px=computePowerbandsEpochs(data=clean_data,epoch_length=30)



stages=extract_data_per_sleep_stage(clean_data[:,:,:-1],clean_hyp,'AASM')



#save_path=path+subject_name
##save_data(stages,save_path,'npy')


#remove artefacted epochs
