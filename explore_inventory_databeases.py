
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
sub_name=[]
for sub in df['Subject_id']:
    pos=sub.find('_')
    sub_name.append(sub[0:pos])

ch_name=[]
nbre_ch=[]

chan=df['EEG channel names']
for ch in chan:
    ch_temp= ch.split(',')
    matching = [s for s in ch_temp if "EEG" in s]
    ch_name.append(matching)
    nbre_ch.append(len(matching))


df['Subject_id']=sub_name
df['EEG channel names']=ch_name
df['Number of EEG channels']=nbre_ch

colm={'Subject_id','EEG channel names','Number of EEG channels'}
df_fin=pd.DataFrame()

#df.to_excel('RECAP_EDF_TOTAL.xlsx')


#import matplotlib.pyplot as plt

#plt.hist(df['Number of EEG channels'],bins=range(23))

a,i=np.histogram(df['Sampling frequency'])
k=np.where(a>0)[0]
ch=a[a>0]
#explode=(0,0.1,0,0,0,0,0,0)
##plt.pie(ch,explode=explode,labels=k,autopct='%1.1f%%',shadow=True)
#for sub in sub_name:
#    for s in df['Subject_id']:
#
#        if sub in s :
#            print s
