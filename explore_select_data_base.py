
import pandas as pd
import numpy as np
import os
from shutil import copyfile

#--------------------------------------------------------#
#                        Load data                       #
#--------------------------------------------------------#
main_path='/home/tarek/sleepclassification'
data_base_path='/home/tarek/sleepclassification/DB TRIO MONTREAL.xlsx'
df_tot=pd.read_excel(data_base_path)

#--------------------------------------------------------#
#                        Set parameters                  #
#--------------------------------------------------------#

data_to_select='Ctrl'# data to select could be :
                            #'TRIO'  :whole data base
                            #'MCI' : MCI group
                            #'AHI' : AHI group
                            #'SOMANA' :Somana group (subject from julie cohort)
                            #'TBI'    :TBI group
                            #'Ctrl'   : Control group (including SOMANA subjects)

cutoff_age=41       # cutoff parameter on subject age
save_group_db=False # save or not dataframe of sub-group data bases
save_path=os.path.join(main_path,data_to_select+'_db.xlsx')
info_file=os.path.join(main_path,data_to_select+'_infos.txt')
copy_EEG_data=False # get a copy of EEG data of one sub-group
EEG_data_path='/media/tarek/CEAMS_database/TRIO_data_base'
#---------------------------------------------------------#


if data_to_select=='TRIO':
    df=df_tot
elif data_to_select=='MCI':

    df=df_tot.loc[df_tot['MCI']==1]


elif data_to_select=='AHI':
    df=df_tot.loc[df_tot['AHI_Group']==1]
elif data_to_select=='TBI':
    df=df_tot.loc[df_tot['TBI']==1]
elif data_to_select=='Ctrl':
    df=df_tot.loc[(df_tot['MCI']!=1) & (df_tot['AHI_Group']!=1 ) & (df_tot['TBI']!=1) ]
elif data_to_select=='SOMANA':
    df=df_tot.loc[df_tot['Project']=='SOMANA']


if save_group_db:
    print '\n{db} data base saved in: '.format(db=data_to_select), save_path,'\n'
    if not os.path.exists(save_path):
        df.to_excel(save_path)



nb_subj=df.shape[0]

nb_man=df.loc[df['Gender']==1].shape[0]
nb_women=df.loc[df['Gender']==2].shape[0]

df_young_m_all=df.loc[(df['AgeAtPSGrecording']<cutoff_age) & (df['Gender']==1)]
nb_young_m_all=df_young_m_all.shape[0]
df_old_m_all=df.loc[(df['AgeAtPSGrecording']>=cutoff_age) & (df['Gender']==1)]
nb_old_m_all=df_old_m_all.shape[0]
df_young_w_all=df.loc[(df['AgeAtPSGrecording']<cutoff_age) & (df['Gender']==2)]
nb_young_w_all=df_young_w_all.shape[0]
df_old_w_all=df.loc[(df['AgeAtPSGrecording']>=cutoff_age) & (df['Gender']==2)]
nb_old_w_all=df_old_w_all.shape[0]



print '{db} data base all subject age and gender informations : '.format(db=data_to_select),'\n'
print 'Nombre total des sujets: ', nb_subj ,'\n'
print 'Nombre total des hommes: ', nb_man
print '     Nombre total des jeunes hommes: ', nb_young_m_all
print '     Nombre total des hommes ages : ', nb_old_m_all,'\n'
print 'Nombre total des femmes: ', nb_women
print '     Nombre total des jeunes femmes: ', nb_young_w_all
print '     Nombre total des femmes agees : ', nb_old_w_all

if not os.path.exists(info_file):
   with open(info_file, 'w') as f:
        print >> f, '{db} data base all subject age and gender informations : '.format(db=data_to_select),'\n'
        print >> f,'Nombre total des sujets: ', nb_subj ,'\n'
        print >> f,'Nombre total des hommes: ', nb_man
        print >> f,'     Nombre total des jeunes hommes: ', nb_young_m_all
        print >> f,'     Nombre total des hommes ages : ', nb_old_m_all,'\n'
        print >> f,'Nombre total des femmes: ', nb_women
        print >> f,'     Nombre total des jeunes femmes: ', nb_young_w_all
        print >> f,'     Nombre total des femmes agees : ', nb_old_w_all
   f.close()

sig_files,sig_files_path=[],[]
sts_files,sts_files_path=[],[]
if copy_EEG_data==True:
    if data_to_select == 'Ctrl':
        data_folder='Controls'
    elif data_to_select=='MCI' :
        data_folder='MCI'
    elif data_to_select=='TBI':
        data_folder='TBI'
    elif data_to_select=='AHI':
        data_folder='AHI'


    for path, subdirs, files in os.walk(EEG_data_path):
    #    all_files.append(files)
        for name in files:
            if (name.endswith(".SIG")) or (name.endswith(".sig")):
                sig_files.append(name)
                sig_files_path.append(os.path.join(path,name))
            if (name.endswith(".sts")) or (name.endswith(".STS")):
                sts_files.append(name)
                sts_files_path.append(os.path.join(path,name))



    cols=['ParticipantCode','Project','Gender','AgeAtPSGrecording']
    new_df=df[cols]
    subjects_names=new_df['ParticipantCode'].values.tolist()
    Selected_sig_path,Selected_sts_path=[],[]

    for i,a in enumerate(subjects_names):
        for j,sub in enumerate(sig_files):
            if str(a).encode('ascii','ignore') in sub :
                print a,sub
                Selected_sig_path.append(sig_files_path[j])
                #dest_folder='/media/tarek/ADATA HV620/TRIO_db/Patients/{f}/{s}/'.format(f=data_folder,s=a)
                dest_folder='/media/tarek/ADATA HV620/TRIO_db/{f}/SIG_STS/'.format(f=data_folder)

#                if not os.path.exists(dest_folder):
#                    os.mkdir(dest_folder)
#                if not os.path.exists(dest_folder+sub):
                copyfile(sig_files_path[j],dest_folder+sub)
#
    for i,a in enumerate(subjects_names):
        for j,sub1 in enumerate(sts_files):
            if str(a).encode('ascii','ignore') in sub1 :
                print a,sub1
                Selected_sts_path.append(sts_files_path[j])
                #dest_folder='/media/tarek/ADATA HV620/TRIO_db/Patients/{f}/{s}/'.format(f=data_folder,s=a)
                dest_folder='/media/tarek/ADATA HV620/TRIO_db/{f}/SIG_STS/'.format(f=data_folder)

#                if not os.path.exists(dest_folder):
#                    os.mkdir(dest_folder)
                #if not os.path.exists(dest_folder+sub1):
                copyfile(sts_files_path[j],dest_folder+sub1)
    #new_df.to_excel('/media/tarek/ADATA HV620/TRIO_db/Patients/'+data_to_select+'_db.xlsx')
