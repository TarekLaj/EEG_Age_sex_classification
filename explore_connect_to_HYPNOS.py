
#import tempfile
from smb.SMBConnection import SMBConnection
import pandas as pd

conn = SMBConnection('tlajnef', 'TA2018?', 'tar','HYPNOS', use_ntlm_v2 = True)
assert conn.connect('10.129.31.60' , 139)
txt_file='/home/tarek/sleepclassification/list_sig_Jessica.txt'
#smb://hypnos/dossiers%20communs/Chrono-Commun/Jessica/Maitrise/Hommes%20+%20Femmes
#files=conn.listPath('Dossiers communs','/Archives.PSG/'u'Apn\xe9es_MCI_Trauma')
files=conn.listPath('Dossiers communs','/Chrono-Commun/Jessica/Maitrise/Hommes + Femmes')
#files=conn.listPath('Dossiers communs','/Chrono-Commun/Jessica/SIG et STS hormones femmes/Femmes middles')

file_list=[]
for i in range(len(files)):
    #if (files[i].filename.endwith('.SIG') )or (files[i].filename.endwith ('.sig')):
    #if files[i].filename.encode('ascii','ignore').endswith(".SIG") or files[i].filename.encode('ascii','ignore').endswith(".sig") :


    if files[i].filename.encode('ascii','ignore').endswith(".STS") or files[i].filename.encode('ascii','ignore').endswith(".sts"):
        file_list.append(files[i].filename.encode('ascii','ignore'))
#        with open(txt_file, 'w') as f:
#            f.write (files[i].filename.encode('ascii','ignore'))
#            f.write('\n')
#    f.close()
