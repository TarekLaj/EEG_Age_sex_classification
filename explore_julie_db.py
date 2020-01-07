#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:26:31 2017

@author: tarek
"""


import pandas as pd 
import numpy as np 


data_base_path='/home/tarek/sleepclassification/Stats_machine_learning.xls'
data_df=pd.read_excel(data_base_path)

man_df_all=data_df.loc[data_df['Sexe']==1]

nb_man_all=man_df_all.shape[0]


nb_young_m_all=man_df_all.loc[man_df_all['Groupe']==1].shape[0]
nb_old_m_all=man_df_all.loc[man_df_all['Groupe']==2].shape[0]

women_df_all=data_df.loc[data_df['Sexe']==2]

nb_women_all=women_df_all.shape[0]
nb_young_w_all=women_df_all.loc[women_df_all['Groupe']==1].shape[0]
nb_old_w_all=women_df_all.loc[women_df_all['Groupe']==2].shape[0]

print ('nombre total des hommes: ', nb_man_all)
print ('nombre total des jeunes hommes: ', nb_young_m_all)
print ('nombre total des hommes ages : ', nb_old_m_all)

print('nombre total des femmes: ', nb_women_all)
print ('nombre total des jeunes femmes: ', nb_young_w_all)
print ('nombre total des femmes agees : ', nb_old_w_all)


#Ctrl_df=df.loc[(df['MCI']!=1) & (df['AHI_Group']!=1 ) & (df['TBI']!=1) ] #&(df['ReasonsForAPossibleExclusion'].isnull())]
