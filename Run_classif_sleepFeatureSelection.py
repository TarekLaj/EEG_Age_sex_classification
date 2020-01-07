#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                               Librairies
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Import standard libraries
import datetime as dt
import pandas
# Import custom libraries
#from classif_utils import SaveExcel
from classif_utils import RemoveNaNs
from classif_utils import ComputeRatio
from classif_utils import PlotHist
from classif_utils import CreateFolderArchitecture
from classif_utils import GetClasses
from classif_utils import GetSamples

# Allow to display 50 columns while using print() (> needed for the features)
pandas.set_option('display.max_columns', 100)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                               Parameters
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
inputFileName       = "Features_all.xls"

rmNaNsMethod        = 'Subject' # Subject - Feature
clusters            = 'Mix'    # 'Sexe' - 'Groupe' - 'Mix'
standardizationType = 'Across_Subjects' #'None'-'Across_Group'-'Across_Subjects'
classifier          = 'SVC' # LinearSVC
nbOfSplit           = 10  # nb of times to separate dataset into random subset
kFolds              = 10  # nb of folds in cross-validation
nbPermutation       = 1000 # number of time to permute data to test classsifier
maxNbrFeaturesSFFS  = 41 # maximum number of features contained in best set (41)
balance_flag        = 0  # balance old woman and old man number or not
nbrOfTrees          = 50 # number of random binary trees to generate in forest
FS_method           = 'SFFS' # SFFS: sequential forward feature selection
                             # RFECV: Recursive feature elemination
                             # RandForest: Random forest
featureList         = [] # []: execute feature selection
                         # ['Freq Fus', ... ,'Alpha abs']: train estimator with
                         #                                 listed features only
                         # [0.7]: Execute feature selection and then train
                         #        estimator with features occuring more often
                         #        than 70% of the time in best set.
verbose             = 0  # 1: display steps, 0: display only results
debug_flag          = 0  # activate (1) or not (0) some debug messages
permutation_flag    = 1  # 1: use permutation, 0: use binomial law

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                               Bootstrap
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Assign class names
if clusters == 'Groupe':
    clusterNames = ['Jeune','Vieux']
elif clusters == 'Sexe':
    clusterNames = ['Femme','Homme']
elif clusters == 'Mix':
    clusterNames = ['Jeune_Femme','Vieille_Femme','Jeune_Homme','Vieil_Homme']

# Create the result folder if not already existing
currentDateTime, resultDir = CreateFolderArchitecture(FS_method+'_'+clusters+
                                                      '_'+classifier+'_'+
                                                      rmNaNsMethod)

# form string explaining some parameter
resultFileName = resultDir+'results_'+FS_method+'_'+currentDateTime+'.xlsx'
xlSheetName = FS_method+'_'+clusters+'_'+rmNaNsMethod
strPermutation = '\nNumber of permutations\t\t : '+str(nbPermutation)+'\n'
strBalance = 'Balance nbr of old women and men : '
strRandForestParam = '\nNumber of trees\t\t\t : '+str(nbrOfTrees)+'\n'
randForestFlag = True if (FS_method == 'RandForest') else False
if len(featureList) == 0:
    strFeatList = 'No'
elif isinstance(featureList[0], float):
    strFeatList = str(featureList[0])
else:
    strFeatList = str(featureList)

config = 'Start time\t\t\t : '+currentDateTime+'\n'+ \
         'Input file\t\t\t : '+ inputFileName+'\n'+ \
         'Save results to\t\t\t : '+resultFileName+'\n'+ \
         'Sheet name\t\t\t : '+xlSheetName+'\n'+ \
         'NaN removing method\t\t : '+rmNaNsMethod+'\n'+ \
         'Classify by\t\t\t : '+clusters+'\n'+ \
         'Classifier type\t\t\t : '+classifier+'\n'+ \
         'Number of splits\t\t : '+str(nbOfSplit)+'\n'+ \
         'Number of cross-validation folds : '+str(kFolds)+ \
         (strPermutation if permutation_flag else '\n') + \
         'Features Groupeselection method\t : '+FS_method+'\n'+ \
         'Max size of best feature set\t : '+str(maxNbrFeaturesSFFS)+'\n'+ \
         'Limited features\t\t : '+strFeatList+'\n'+ \
         strBalance+ (('Yes') if balance_flag else ('No')) + '\n' + \
         'Standardization \t\t : '+str(standardizationType) + \
         (strRandForestParam if randForestFlag else '\n') + \
         'Verbose \t\t\t : '+ ('Yes' if verbose else 'No') +'\n'+ \
         'Debug message \t\t\t : ' + ('Yes' if debug_flag else 'No') +'\n\n'

# Save configuration in .txt file
with open (resultDir+'config.txt','a') as f : f.write(config)
# display all parameters in terminal
print(config)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                                   Prepare data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# read excel file : header=0 to read and be able toGroupe use header names
rawData = pandas.read_excel(inputFileName,header=0)

# Remove subjects or attributes containing NaNs according to rmNaNsMethod
nanFreeData, removedData = RemoveNaNs(rawData,rmNaNsMethod,debug_flag)
# Select only classes from dataset
y = GetClasses(nanFreeData, clusters)
# Select only samples from dataset
samples = GetSamples(nanFreeData)
# Compute some power ratios
samples = ComputeRatio(samples)
# Create a list of the feature's names
featureNames = list(samples)
if verbose:
    print('\tComplete\n')

# Create the estimator and RFE object with a cross-validated score.
#if clusters == 'Mix':
if classifier == 'LinearSVC':
    from sklearn.svm import LinearSVC
    clf = LinearSVC(dual=False,multi_class='ovr')
elif classifier == 'SVC':
    from sklearn.svm import SVC
    clf = SVC(kernel="linear",shrinking=False)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                               Feature selection
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
if verbose:
    print('Executing feature selection')
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                                   RFECV
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
if FS_method=='RFECV':
    from classif_utils import ExecuteRFECV
    excelResults = ExecuteRFECV(samples,y,featureNames,featureList,
                                clusters,clusterNames,
                                clf,kFolds,nbOfSplit,
                                standardizationType,removedData,
                                permutation_flag,nbPermutation,
                                balance_flag,
                                currentDateTime,resultDir,
                                debug_flag,verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                                   SFFS
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
elif FS_method=='SFFS':
    from classif_utils import ExecuteSFFS

    excelResults = ExecuteSFFS(samples,y,featureNames,featureList,
                               clusters,clusterNames,
                               clf,kFolds,nbOfSplit,
                               maxNbrFeaturesSFFS,
                               standardizationType,removedData,
                               permutation_flag,nbPermutation,
                               balance_flag,
                               currentDateTime,resultDir,
                               debug_flag,verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                                   Random Forest
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
elif FS_method=='RandForest':
    from classif_utils import ExecuteRandForest
    excelResults = ExecuteRandForest(samples,y,featureNames,
                                     clusters,clusterNames,
                                     clf,kFolds,nbOfSplit,nbrOfTrees,
                                     standardizationType,removedData,
                                     permutation_flag,nbPermutation,
                                     currentDateTime,resultDir,
                                     debug_flag,verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                                   Random Forest
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
elif FS_method=='None':
    from classif_utils import ExecuteClassifier
    excelResults = ExecuteClassifier(samples,y,featureList,
                                     clusters,clusterNames,
                                     clf,kFolds,nbOfSplit,nbrOfTrees,
                                     standardizationType,removedData,
                                     permutation_flag,nbPermutation,
                                     currentDateTime,resultDir,
                                     debug_flag,verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#                               SAVE RESULTS
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print(str(dt.datetime.now().strftime("%Y_%b_%d_%H-%M-%S")))
print('\tComplete\n')
#saveTo = resultDir+'results_'+FS_method+'_'+currentDateTime+'.xlsx'
excelResults.to_excel(resultFileName,sheet_name=xlSheetName)

if len(featureList)==0 or isinstance(featureList[0],float):
    # plot features occurence histogram
    savedPlotName = resultDir+'BestFeatures_hist_Sexe.png'+str(nbOfSplit)+'.png'
    title = 'Best features occurence '+currentDateTime
    PlotHist(excelResults['Occurence_Best'],
             featureNames,
             title,
             savedPlotName)
