
import pandas as pd
import numpy as np
import os
from shutil import copyfile,move

sexe=1
age=2
if sexe==1:
    sexe_folder='Women'
else:
    sexe_folder='Men'
if age==1:
    age_folder='Young'
else:
    age_folder='Middle_aged'

file_xls=os.path.join('/media/tarek/CEAMS_database/data_EEG_sleep/Jessica_db','liste_depistage.xlsx')
df_tot=pd.read_excel(file_xls)
subjects_names=df_tot['Subjects_ids'].values.tolist()

path='/media/tarek/CEAMS_database/data_EEG_sleep/Jessica_db/{gender}/{age}/'.format(gender=sexe_folder,age=age_folder)
#EDF_path='/media/tarek/ADATA HV620/Jessica_db/Men/Middle_aged/EDF/'

STS_files=[]
XML_files=[]
#subj_names=[]
EDF_files=[]
SIG_files=[]
EDF_f=[]
XML_f=[]
SIG_f=[]
STS_f=[]
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".edf"):
            EDF_f.append(filename)
            EDF_files.append(root+'/'+filename)
        if filename.endswith(".xml"):
            XML_f.append(filename)
            XML_files.append(root+'/'+ filename)

        if (filename.endswith(".SIG")) or (filename.endswith(".sig")):
            SIG_files.append(root+filename)
            #subj_names.append(filename[:-4])
            SIG_f.append(filename)
        if (filename.endswith(".sts")) or (filename.endswith(".STS")):
            STS_f.append(filename)
            STS_files.append(root+filename)

file_to_cp=STS_files
f=STS_f
#for j,sub in enumerate(file_to_cp):
#    for i,a in enumerate(subjects_names):
#        dest_folder=os.path.join(path,a[:-4])
#        if not os.path.exists(dest_folder):
#            os.makedirs(dest_folder)
#       # print(sub,a)
#        if a.lower() in sub.lower():
#            move(sub,os.path.join(dest_folder,f[j]))
#            print(os.path.join(dest_folder,f[j]))
#        matching = [s for s in file_to_cp if subj in s]
#            df_tot['Sts_Path'][i]=sts_files_path[j]
##            if (df_tot['Sexe'][i]==1) & (df_tot['Groupe age'][i]==1):
#                copyfile(path,'/media/tarek/ADATA HV620/Jessica_db/Women/Young/{s}'.format(s=sub))
        #print (df_tot.loc[df_tot['Subjects_ids'] [i][:-4]== sub].iloc[0])
    #move(matching[0],'/media/tarek/ADATA HV620/Jessica_db/Men/Middle_aged/EDF/')

for i,a in enumerate(subjects_names):
        for j,sub in enumerate(f):
            if str(a).encode('ascii','ignore')[:-4] in sub.decode('utf-8'):
                print a,sub
                dest_folder=path+'/{s}/'.format(s=a[:-4])
                #dest_folder='/media/tarek/ADATA HV620/TRIO_db/{f}/SIG_STS/'.format(f=data_folder)
                if not os.path.exists(dest_folder):
                    os.mkdir(dest_folder)
#                if not os.path.exists(dest_folder+sub):
                    print(file_to_cp[j])
                    print('*****************')
                    print(dest_folder+sub)
                move(file_to_cp[j],dest_folder+sub.decode('utf-8'))
