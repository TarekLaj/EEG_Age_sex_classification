import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from brainpipe.classification import *
from brainpipe.visual import *
import mne
from mne.viz import plot_topomap
from mne import pick_types, find_layout
from scipy import stats
from random import sample
import os


def PlotBestCombinationMetrics(data,figName):
    # Plot mean/std
    plt.figure(dpi=300)
    plt.title('Metrique choix meilleure combinaison')
    plt.xlabel("nb attributs dans combinaison")
    plt.xticks(range(data.shape[0]))
    plt.ylabel("Moyenne sur ecart-type")
    for n in list(range(data.shape[1])):
        plt.plot(list(range(1,data.shape[0]+1)),data.iloc[:,n])
    plt.savefig(figName, bbox_inches='tight')
    plt.clf()
    plt.close()

def PlotPermHist(data,testAvgDA,dateTime,savedPlotName):
    from math import floor

    if len(np.shape(data)) == 2:
        df = pandas.DataFrame(data=data)
        dataHist, bins = np.histogram(df,bins=100,range=(0,1))
        pValue = dataHist[floor(testAvgDA*100):].sum()/dataHist.sum()

    elif len(np.shape(data)) > 2: # sum histograms bins for each epoch
        df = pandas.DataFrame(data=data)
        df = df.T
        dataHist = pandas.DataFrame()
        for i in list(range(len(df))):
            tmp, bins = np.histogram(df.iloc[i][:],bins=100,range=(0,1))
            dataHist = tmp if i == 0 else dataHist + tmp

        pValue = dataHist[floor(testAvgDA.iloc[0][1]*100),:].sum() / \
                    dataHist.sum()

    print('\nPermutation pValue : {}'.format(pValue))

    plt.figure(figsize=[10,8],dpi=300)
    plt.title('Permutation decoding accuracy\nN={} pValue = {}\n {}'.format(
            dataHist.sum(),pValue,dateTime))
    plt.xlabel("Decoding accuracy")
    plt.ylabel("Number of occurence")
    plt.bar(range(100),dataHist,width=0.9,label='Occurances',align='center')
    plt.axvline(x=testAvgDA*100,color='r',label='test DA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedPlotName, bbox_inches='tight')
    plt.clf()
    print('\tComplete')
    return plt


def PlotHist(dat,xLabels,title,savedPlotName):#,maxY

    manualFlag = 0
    # for manual use
    if manualFlag:
        dat = pandas.read_excel('filename.xlsx',header=0)
        occ = dat['Occurence_Best']
        names = dat['Features_Name']
        data = occ.dropna()
    else:
        data = cp.deepcopy(dat).dropna()
        labels = pandas.DataFrame(data=xLabels)
    
    data.sort_values(ascending=False,inplace=True)

    if manualFlag:
        xTickLabel = names[data.index]
    else:
        tmp = labels.iloc[data.index]
        xTickLabel = tmp

    plt.figure(dpi=300)
    plt.title(title)
    plt.xlabel("Features")
    plt.xticks(range(len(data)),xTickLabel,rotation='vertical')
    plt.ylabel("Occurence")
    plt.bar(range(len(data)),data,width=0.9,label='Occurences',align='center')
    plt.legend()
    plt.savefig(savedPlotName, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          precision=8,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],precision),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    return plt

def GetSubsetOccurence(bestSets):
    """
    Ne depasse pas les combinaisons de 8 attributs, a 9 la memoire bust
    """
    import itertools
#    import numpy as np
    import pandas as pd
    from gc import collect as clr


    # find unique values of all bestSets and get maximum set size
    unique = set()
    maxSetSize = 0
    for i in bestSets:
        unique.update(set(i))
        if len(i) > maxSetSize:
            maxSetSize = len(i)
    print('Number of unique elements found = {}'.format(len(unique)))

    # create a list of all possible combinations with unique values
    hist    = pd.DataFrame()
    for j in list(range(1,maxSetSize+1)):
        print('Searching for all subsets of '+str(j)+' elements...')
        subset = list(itertools.combinations(unique,j))
        print('    {} subsets found'.format(len(subset)))
        print('Computing histogram...')
        df=[] # list to contain histogram of subsets of size j and occurences
        for k in subset:
            df.append([k,0])
            for l in bestSets:
                if set(k).issubset(l):
                    df[-1][1] += 1
            if df[-1][1] == 0:
                del df[-1]
        df1 = pd.DataFrame(data=df,
                           columns=['Subset_'+str(j),'Occurence_'+str(j)])
#        hist = pd.concat([hist,df1],axis=1)

        print('Saving histogram to excel...')
        df1.to_excel('subset_'+str(j)+'.xlsx')

        from slpClass_toolbox import PlotHist
        PlotHist(df1[1],df1[0],'Subsets occurences','Comb_Hist_'+str(j)+'.png')
#        hist.to_excel('subset_'+str(j)+'.xlsx')
        clr() # remove when finish testing

    

    return hist
def Topoplot_DA(DA=None,sensors_pos=None,masked=False,chance_level=0.05,Da_perm=None,Da_bino=None,figures_save_file=None,show_fig=False):
    if masked==True:
        mask_default = np.full((len(DA)), False, dtype=bool)
        mask = np.array(mask_default)
        if Da_perm is not None:
            nperm=Da_perm.shape[0]
            indx_chance=int(nperm-chance_level*nperm)
            daperm=np.sort(Da_perm,axis=0)
            mask[DA>=np.amax(daperm[indx_chance-2,:])] = True
        if Da_bino is not None:
            mask[DA>=Da_bino] = True



        mask_params = dict(marker='*', markerfacecolor='w', markersize=15)
        fig1 = plt.figure(figsize = (10,15))
        ax1,_ = plot_topomap(DA, sensors_pos,
                             cmap='inferno', show=show_fig,
                             vmin=0, vmax=100,
                             mask = mask,
                             mask_params = mask_params,outlines='skirt')
                             #names=elect,show_names=True)
    elif masked==False:


        fig1 = plt.figure(figsize = (10,15))
        ax1,_ = plot_topomap(DA, sensors_pos,
                             cmap='inferno', show=show_fig,
                             vmin=0, vmax=100,outlines='skirt')
    
    if figures_save_file is not None:
        plt.savefig(figures_save_file, dpi = 300)
