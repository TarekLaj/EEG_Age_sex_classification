import collections
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
clf_name='LDA'
cds=['c1','c2']
cols=['col2']
stades=['AWA','S1','S2','SWS','Rem']

for col in cols:
    for C in cds:
        for stade in stades:
            DA_path='/home/karim/results_C1C2/Resultats_feat_select/{cl}/MultiFeatures/MultiFeatures'.format(cl=col)
            DA_file_format='DA_{clf}_SFS_{c}_{cl}_{stade}'.format(clf=clf_name,c=C,cl=col,stade=stade)
            feat_file='Selected_features_{clf}_{c}_{cl}_{stade}'.format(clf=clf_name,c=C,cl=col,stade=stade)
            Da=np.array([])
            f_list,feat_list=[],[]
            for file in os.listdir(DA_path):
                if file.startswith(DA_file_format):
                    da=np.load(os.path.join(DA_path, file))
                    Da=np.hstack((Da,da)) if Da.size else da
                if file.startswith(feat_file):
                    ft=np.load(os.path.join(DA_path, file))
                    f1 = [item for f in ft for item in f]
                    f_list.append(ft)
                    feat_list=feat_list+f1

            print(np.mean(Da))
            compt=collections.Counter(feat_list)

            fig = plt.figure(figsize = (10,10))
            sel_list=list(compt.keys())

            n_time_list=list(compt.values())


            idx=list(np.where(np.array(n_time_list)>5)[0])

            sel_list_fin = [sel_list[index] for index in idx]
            n_time_list_fin=[n_time_list[index] for index in idx]


            L = [x for _,x in sorted(zip(n_time_list_fin, sel_list_fin))]
            T = [x for x,_ in sorted(zip(n_time_list_fin, sel_list_fin))]
            plt.bar(range(len(L)),T)

            plt.xticks(range(len(L)),L)
                #plt.autoscale()
                #ax.set_xticklabels(compt.keys())
            plt.ylim([0,100])
            plt.xticks(rotation=90)


            save_hist='/home/karim/results_C1C2/figures/SFS_hist_{clf}_{c}_{p}_{st}.png'.format(clf=clf_name,
                                                                                      c=C,st=stade,
                                                                                      p=col)
            plt.savefig(save_hist, dpi = 300)


            sensors_pos = sio.loadmat('/home/karim/MATLAB/Work/Projet C1 C2/Coord_2D_Slp_EEG.mat')['Cor']

            ch_names_info =['Fz','Cz','Pz','C3','C4','T3', 'T4', 'Fp1',
                      'Fp2', 'O1', 'O2', 'F3','F4', 'P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
            ntimes=[]
            for ch in ch_names_info:
                if ch in sel_list_fin:
                    ntimes.append(n_time_list_fin[sel_list_fin.index(ch)])
                else:
                    ntimes.append(0)


            freq=np.array(ntimes)

            fig = plt.figure(figsize = (10,5))
            ax,_ = plot_topomap(freq, sensors_pos,names=ch_names_info,show_names=True,cmap='inferno',show=False,
            vmin=0,vmax=100,contours=True)
            fig.colorbar(ax)
            save_topo='/home/karim/results_C1C2/figures/SFS_topo_{clf}_{c}_{p}_{stade}.png'.format(clf=clf_name,
                                                                                      c=C,stade=stade,
                                                                                      p=col)
            plt.savefig(save_topo, dpi = 300)
