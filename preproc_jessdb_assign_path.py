import pandas as pd
import os


usr='tarek'
path_to_dd=os.path.join('/media',usr,'ADATA HV620')
if not os.path.ismount(path_to_dd) :
    raise Exception('Hard drive is not mounted!')
file_xls=os.path.join(path_to_dd,'Jessica_db','liste_depistage.xlsx')

df_tot=pd.read_excel(file_xls)
subjects_names=df_tot['Subjects_ids'].values.tolist()

data_path='/media/tarek/CEAMS_database/Jessica_db/Hommes + Femmes/'

sig_files,files_path,sts_files,sts_files_path=[],[],[],[]
for path, subdirs, files in os.walk(data_path):
#    all_files.append(files)
    for name in files:
        if (name.endswith(".STS")) or (name.endswith(".sts")) :
            sts_files.append(name)
            sts_files_path.append(os.path.join(path,name))
        if (name.endswith(".SIG")) or (name.endswith(".sig")) :
            sig_files.append(name)
            files_path.append(os.path.join(path,name))
df_tot['Subjects_ids']=df_tot['Subjects_ids'].astype(str)
for j,sub in enumerate(sig_files):
    for i,a in enumerate(subjects_names):
        if sub[:-4].lower() in  df_tot['Subjects_ids'][i].lower():
            df_tot['Sig_Path'][i]=files_path[j]

for j,sub in enumerate(sts_files):
    for i,a in enumerate(subjects_names):
        if sub[:-4].lower() in  df_tot['Subjects_ids'][i].lower():
            df_tot['Sts_Path'][i]=sts_files_path[j]
#            if (df_tot['Sexe'][i]==1) & (df_tot['Groupe age'][i]==1):
#                copyfile(path,'/media/tarek/ADATA HV620/Jessica_db/Women/Young/{s}'.format(s=sub))
        #print (df_tot.loc[df_tot['Subjects_ids'] [i][:-4]== sub].iloc[0])
df_tot.to_excel(file_xls)
    #DD= df_tot.where(df_tot['Subjects_ids'][i][:-4]==sub[:-4])c
