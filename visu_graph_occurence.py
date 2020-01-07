#==============================================================================
#                        Plot best sets occurence histogram
#==============================================================================
"""
Plot feature occurence in best combination as an histogram
Excel file used : output from sleepClassification.py in result folder
"""
import pandas
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from eeg_agegenderclassification.visualisation import
fileName = 'results_group.xls'
# read excel file : header=0 to read and be able to use header names
rawData = pandas.read_excel(fileName,header=0)
x = rawData['Features_Name'][:41]
y = rawData['Occurence_Best'][:41]
print(x)
print(y)

plt.figure(dpi=300)
plt.title('Occurence dans les meilleures combinaisons')
plt.xlabel("Attributs")
plt.xticks(range(len(x)),x,rotation='vertical')
plt.ylabel("Occurence")
plt.bar(range(len(y)),y,width=0.9,label='Occurences',align='center')
plt.savefig('Group_Best_Features_Occurence', bbox_inches='tight')
plt.show()

#==============================================================================
#                     Plot best set subsets occurence histogram
#==============================================================================
"""
Plot feature subsets occurence in best combination as an histogram
Excel file used :   Contains at least 2 columns (Subsets_n, Occurence_n)
                    _n stand for number of feature in corresponding subset
                    Note : it should be created manually after running
                    GetSubsetOccurence(bestSets) and aggregating the result
                    files into 1
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'figure.max_open_warning': 0})

fileName = 'Subset_Comparison_10.xlsx'
size = [2,3,4,5,6,7,8] # subset size must be >1 and <= max computed in fileName
treshold = 6 # minimal occurences kept

# read excel file : header=0 to read and be able to use header names
print('Reading excel file...')
rawData = pandas.read_excel(fileName,header=0)
print('    done')

print('Selecting only subsets with occurance higher or egal to '+str(size))
samples = []
hist    = []
for i in size:
    x = rawData['Subset_'+str(i)]
    y = rawData['Occurence_'+str(i)]

    samples = np.concatenate([samples,x[y>=treshold]])
    hist    = np.concatenate([hist,y[y>=treshold]])
print('    done')

# zero-free histogram computation
notZeroIdx = hist.nonzero()
histZeroFree = hist[notZeroIdx]
samplesZeroFree = samples[notZeroIdx]
# nan-free histogram computation
notNanIdx = ~np.isnan(histZeroFree)
cleanedHist = histZeroFree[notNanIdx]
cleanedSample = samplesZeroFree[notNanIdx]

maxOccur = max(cleanedHist)

# Plot and save result
from time import gmtime, strftime
import os
import shutil
d = strftime("%Y-%m-%d %H:%M:%S", gmtime())
os.mkdir(d)
os.mkdir(d+'/Input')
outDir = d+'/Output'+str(size)
os.mkdir(outDir)

shutil.copy2(fileName, d+'/Input/'+fileName)

print('ploting histogram...')
plt.figure(dpi=300)
plt.title('Combinaisons de '+str(size)+' avec occurence >='+str(treshold))
plt.xlabel("Combinations")
if len(cleanedSample) <= 40:
    plt.xticks(range(len(cleanedSample)),cleanedSample,rotation='vertical')
plt.ylabel("Occurence")
plt.bar(list(range(len(cleanedHist))),cleanedHist,label='Occurences',align='center')
plt.savefig(outDir+'/Occurence_Combination_'+str(size), bbox_inches='tight')
plt.grid()
plt.show()

# Save plotted data in excel file
df1 = pandas.DataFrame(data=cleanedSample,columns=['Subset'])
df2 = pandas.DataFrame(data=cleanedHist,columns=['Occurences'])
df3 = pandas.DataFrame(data=[maxOccur],columns=['maximum occurence'])
df4 = pandas.DataFrame(data=cleanedSample[cleanedHist==maxOccur],
                            columns=['subset with max occurence'])
df = pandas.concat([df1,df2,df3,df4],axis=1)
fName = outDir+'/Histogram_data.xlsx'
df.to_excel(fName)

#==============================================================================
#                          Plot best set metric
#==============================================================================
"""
Transform cross-validation learning curves of the SFFS best feature set of
every iteration made into (curve - mean) / std. Stack all curves into a single
plot

    Inputs : result excel file from running sleepFeatureSelection.py

    Output : Graph stacking the new metric for every best feature set. Graph is
             saved as variable outputFileName define it
"""
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# Plot mean/std
fileName = 'results.xlsx'
nbFeat   = 42 # feature number +2
nbIt     = 10 # iteration number during SFFS execution
outputFileName  = 'SFFS_Metric_10_v2'

raw = pd.read_excel(fileName,header=0)
data = pd.DataFrame()

# Get average DA and standard deviation from file and compute mean over std
for k in list(range(nbIt)):
    data = pd.concat([data,
                      raw['avg_score_'+str(k)][1:nbFeat],
                      raw['std_dev_'+str(k)][1:nbFeat]],
                    axis=1)
    tmp = pd.DataFrame(data=data['avg_score_'+str(k)] / data['std_dev_'+str(k)],
                       columns=['avg_over_std_'+str(k)])
    data = pd.concat([data,tmp],axis=1)

# plot separated iterations
plt.figure(dpi=300)
plt.title('Moyenne sur ecart-type de la courbe de CV\n N='+str(nbIt))
plt.xlabel("nb attributs dans la combinaison")
plt.xticks(range(nbFeat))
#plt.xlabel(list(range(np.cast['int'](np.round(nbFeat/2)))))
plt.ylabel("Moyenne sur ecart-type")

for n in list(range(nbIt)):
    plt.plot(list(range(1,nbFeat)),data['avg_over_std_'+str(n)])
ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::2])
plt.savefig(outputFileName+'_separeted', bbox_inches='tight')
plt.show()

# plot average of mean over std
s=0
for i in list(range(nbIt)):
    s = s + data['avg_over_std_'+str(i)]
m = s/nbIt

plt.figure(dpi=300)
plt.title('Moyenne sur ecart-type de la courbe de CV\n N='+str(nbIt))
plt.xlabel("nb attributs dans la combinaison")
plt.xticks(range(nbFeat))
plt.ylabel("Moyenne sur ecart-type")
plt.plot(list(range(1,nbFeat)),m)
ax = plt.gca()
ax.set_xticks(ax.get_xticks()[::2])
plt.savefig(outputFileName+'_averaged', bbox_inches='tight')
plt.show()
