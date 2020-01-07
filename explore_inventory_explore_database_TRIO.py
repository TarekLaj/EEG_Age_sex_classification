
import pandas as pd
import numpy as np


data_base_path='/home/karim/sleepclassification/DB TRIO MONTREAL.xlsx'
df=pd.read_excel(data_base_path)
Ctrl_df=df.loc[(df['MCI']!=1) & (df['AHI_Group']!=1 ) & (df['TBI']!=1) ] #&(df['ReasonsForAPossibleExclusion'].isnull())]

Gend_nan_indx=np.where(Ctrl_df['Gender'].isnull()==True)
Ctrl_df=Ctrl_df.drop(Ctrl_df.index[Gend_nan_indx])
# all data
Ctrl_df.to_csv('/home/karim/sleepclassification/Ctrl_data.csv')
df_man=Ctrl_df.loc[Ctrl_df['Gender']==1]
#ag= df_man['AgeAtPSGrecording']

nb_woman_all=Ctrl_df.loc[Ctrl_df['Gender']==2].shape[0]
nb_man_all=Ctrl_df.loc[Ctrl_df['Gender']==1].shape[0]
#Ctrl_df=Ctrl_MCI.loc[Ctrl_MCI['AHI_Group']==0]

df_young_m_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']<41) & (Ctrl_df['Gender']==1)]
nb_young_m_all=df_young_m_all.shape[0]

df_old_m_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']>=41) & (Ctrl_df['Gender']==1)]
nb_old_m_all=df_old_m_all.shape[0]


df_young_w_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']<41) & (Ctrl_df['Gender']==2)]
nb_young_w_all=df_young_w_all.shape[0]

df_old_w_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']>=41) & (Ctrl_df['Gender']==2)]
nb_old_w_all=df_old_w_all.shape[0]

##data with subjective measures
nan_indx=np.where(Ctrl_df['PSQITotalScore'].isnull()==True)
database=Ctrl_df.drop(Ctrl_df.index[nan_indx])

nb_woman=database.loc[database['Gender']==2].shape[0]
nb_man=database.loc[database['Gender']==1].shape[0]
#Ctrl_df=Ctrl_MCI.loc[Ctrl_MCI['AHI_Group']==0]

df_young_m=database.loc[(database['AgeAtPSGrecording']<41) & (database['Gender']==1)]
nb_young_m=df_young_m.shape[0]

df_old_m=database.loc[(database['AgeAtPSGrecording']>41) & (database['Gender']==1)]
nb_old_m=df_old_m.shape[0]


df_young_w=database.loc[(database['AgeAtPSGrecording']<41) & (database['Gender']==2)]
nb_young_w=df_young_w.shape[0]

df_old_w=database.loc[(database['AgeAtPSGrecording']>41) & (database['Gender']==2)]
nb_old_w=df_old_w.shape[0]

#data_CTRL=

print ('with subjectives data \n: ')
print ('nombre total des hommes: ', nb_man)
print ('nombre total des jeunes hommes: ', nb_young_m)
print ('nombre total des hommes ages : ', nb_old_m)

print('nombre total des femmes: ', nb_woman)
print ('nombre total des jeunes femmes: ', nb_young_w)
print ('nombre total des femmes agees : ', nb_old_w)
