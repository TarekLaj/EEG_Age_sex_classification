
# script to read csv recap files : inventaire base de données sommeil

import pandas as pd
import numpy as np
import os

data_list=['Jessica_db_Men', 'Jessica_db_Women'] #,'TRIO_db_Patients','TRIO_db_Controls'

csv_path='/home/tarek/sleepclassification/'
df=[]
data=[]
for data_bd in data_list:
    csv_file=os.path.join(csv_path,'recap_edf_{dd}.csv'.format(dd=data_bd))
    df_temp=pd.read_csv(csv_file)
    df1 = df_temp.iloc[:,1:5].copy()
    df.append(df1)
    data.append(df1)
    print(df_temp.shape)
df = pd.concat(df, axis=0,ignore_index=True)
sub_name_julie=[]
for sub in df['Subject_id']:
    pos=sub.find('_')
    sub_name_julie.append(sub[0:pos])



data_list=['TRIO_db_Controls'] #,'TRIO_db_Patients','TRIO_db_Controls'

csv_path='/home/tarek/sleepclassification/'
df_tr=[]
for data_bd in data_list:
    csv_file=os.path.join(csv_path,'recap_edf_{dd}.csv'.format(dd=data_bd))
    df_temp_tr=pd.read_csv(csv_file)
    df1_tr = df_temp_tr.iloc[:,1:5].copy()
    df_tr.append(df1_tr)
    print(df_temp_tr.shape)
df_tr = pd.concat(df_tr, axis=0,ignore_index=True)
sub_name_Trio=[]
for sub in df_tr['Subject_id']:
    pos=sub.find('_')
    sub_name_Trio.append(sub[0:pos])
commun_list=[]
for sub in sub_name_Trio:
    for s in sub_name_julie:

        if sub in s :
            commun_list.append(s)
