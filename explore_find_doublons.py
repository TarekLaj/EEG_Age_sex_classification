
import pandas as pd


#import collections

file1='/home/karim/eeg_agegenderclassification/Final_List.xlsx'

df1=pd.read_excel(file1)
subj=df1['Subject_id'].tolist()
list_all=df1['Subject_id'].tolist()
seen = set()
doublons,idx_rem=[],[]
for idx, x in enumerate(subj):

    list_all.remove(x)
    for j in list_all:

        if x in j:
           # print ('x',x,'j',j, 'idx',idx)
            #df1.drop([idx])x
            idx_rem.append(idx)
#        doublons.append(x)
#        seen.add(x)


df2=df1.drop(idx_rem) #data frame contaning data without redendence

number_of_women=df2.loc[df2['Gender']==2].shape[0]
number_of_men=df2.loc[df2['Gender']==1].shape[0]
