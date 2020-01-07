
import pandas as pd
file_xls='/home/tarek/sleepclassification/list subject jessica.xlsx'

df_tot=pd.read_excel(file_xls)

sig_list_temp=df_tot['SIG Files'].tolist()
sts_list_temp=df_tot['STS files'].tolist()
middle_femme_list_temp=df_tot['middle femme']
jeune_femme_list_temp=df_tot['jeune_femme']

sig_list,sts_list,middle_femme_list,jeune_femme_list=[],[],[],[]



for x in sig_list_temp:
    sig_list.append(x[:-4])


for y in sts_list_temp:
    if str(y) != 'nan':
        sts_list.append(y[:-4])

for y in middle_femme_list_temp:
    if str(y) != 'nan':
        middle_femme_list.append(y[:-4])

for y in jeune_femme_list_temp:
    if str(y) != 'nan':
        jeune_femme_list.append(y[:-4])


sujet_sans_STS_list=list(set(sig_list) - set(sts_list))

STS_jeune_femmes=list(set(sig_list) - set(sts_list))

#sujet_sans_STS_list=[]
#for j in sig_list:
#    for i in sts_list:
#        if j != i:
#            sujet_sans_STS_list.append(j)
