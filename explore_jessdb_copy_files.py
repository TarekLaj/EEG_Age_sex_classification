
from shutil import copyfile
import os
import pandas as pd

usr='tarek'
path_to_dd=os.path.join('/media',usr,'ADATA HV620')
if not os.path.ismount(path_to_dd) :
    raise Exception('Please Plug ADATA HV620 hard drive !')
file_xls=os.path.join(path_to_dd,'Jessica_db','liste_depistage.xlsx')

df_tot=pd.read_excel(file_xls)

sexe=2
age=1
if sexe==1:
    sexe_folder='Women'
else:
    sexe_folder='Men'
if age==1:
    age_folder='Young'
else:
    age_folder='Middle_aged'


selected_df=df_tot.loc[(df_tot['Sexe']==sexe) & (df_tot['Groupe age']==age)]
dest_file='/media/tarek/ADATA HV620/Jessica_db/{s}/{a}/'.format(s=sexe_folder,a=age_folder)

for index, row in selected_df.iterrows():
    print 'copy files ', row['Sig_Path']
    #dest_file='/media/tarek/ADATA HV620/Jessica_db/{s}/{a}/{sub}.sig'.format(s=sexe_folder,a=age_folder,sub=row['Subjects_ids'][:-4])
    #print dest_file+row['Subjects_ids'][:-4]+'.sig'
    copyfile(row['Sig_Path'],dest_file+row['Subjects_ids'][:-4]+'.sig')
    copyfile(row['Sts_Path'],dest_file+row['Subjects_ids'][:-4]+'.sts')
