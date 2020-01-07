import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import permutation_test_score
import scipy.io as sio
#import matplotlib.pyplot as plt
#from mne.viz import plot_topomap
#from mne import pick_types, find_layout
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as pltscipy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def get_data(list1=[],list2=[],group=[],data_path=None,C=[],stade=None,moy=0,p=2,add_pe=False):
    Dr_data=[]
    nDr_data=[]
    y=[]
    for s1,s2 in zip (range(list1.shape[0]),range(list2.shape[0])):
        M1=load_C1C2(subject=list1[s1],C=C,stade=stade,moy=moy,path=data_path,p=p)
        Dr_data.append(M1)
        M2=load_C1C2(subject=list2[s2],C=C,stade=stade,moy=moy,path=data_path,p=p)
        nDr_data.append(M2)

        #y.append([1]*len(Dr_data)+[0]*len(nDr_data))

    if add_pe:
        pndr,pdr=np.array([]),np.array([])
        print('loading Permutation entropy: ')
        for s1,s2 in zip(range(sDreamer.shape[0]),range(sNnDreamer.shape[0])):
            p1=load_pe(path=pe_path,subject=sDreamer[s1],stade=st,moy=moy)
            pdr=np.concatenate((pdr,p1),axis=0) if pdr.size else p1
            p2=load_pe(path=pe_path,subject=sNnDreamer[s2],stade=st,moy=moy)
            pndr=np.concatenate((pndr,p2),axis=0) if pndr.size else p2
        xpe=np.vstack((pdr,pndr))

    X=Dr_data+nDr_data

    y_ndr=[np.asarray([0]*sub.shape[0]) for sub in  nDr_data]
    y_dr=[np.asarray([1]*sub.shape[0]) for sub in  Dr_data]
    y=y_dr+y_ndr

    groups=[np.asarray([group[ind]]*sub.shape[0]) for ind,sub in enumerate(X)]

    sz_ndr = [sub.shape[0] for sub in  nDr_data]
    sz_dr = [sub.shape[0] for sub in  Dr_data]
    sizes=np.concatenate((sz_dr,sz_ndr),axis=0)
    return X,y,groups,sizes

def load_C1C2(subject=[],C=None,stade=None,moy=1,path=None,p=2):

    file='/lin_C1C2C3_guy{s}_{st}_{cl}.mat'
    #file = '/{c}/C1C2C3_S{s}_{st}_{c}.mat'
    ck = sio.loadmat(path+file.format(s=subject,st=stade,cl=p))[C][0:19,:]
    if moy:
        mat=np.mean(ck,axis=1)
        M=np.reshape(mat,(1,19))
    else:
        M=ck.T

    return np.array(M)
def load_pe(path=None,subject=[],stade=None,moy=0):
    file='permut_s{}.mat'.format(str(subject))
    M=sio.loadmat(path+file)['permut_{}'.format(stade)][:,0:19]
    if moy:
        M=np.reshape(np.mean(M,axis=0),(1,19))
    return M

def get_classifier(clf_name=None, inner_cv=None):
    clf = None
    fit_params = {}
    if clf_name=='logreg':
        if inner_cv==None:
            clf_init = LogisticRegression(random_state=0)
            clf=make_pipeline(StandardScaler(),clf_init)
        else:
            clf_init=LogisticRegression(random_state=0,solver='liblinear')

            random_grid = {'penalty':['l1','l2'] ,
                           'C': np.logspace(-4, 4, 20)}
            n_iter_search=40

            Rnd_Srch = RandomizedSearchCV(clf_init, param_distributions=random_grid,
                                           n_iter=n_iter_search, cv=inner_cv,iid=True)
            clf=make_pipeline(RobustScaler(),Rnd_Srch)

    if clf_name=='LDA':
        clf=LDA()

    if clf_name=='RBF_svm':
        svm=SVC(kernel='rbf')
        # parameters for grid search
        if inner_cv==None:
            clf=svm

        else:
            p_grid = {}
            p_grid['gamma']= [1e-3, 1e-4]
            p_grid['C']= [1, 10, 100, 1000]
            # classifier
            #clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            n_iter_search=10
            Rnd_Srch = RandomizedSearchCV(svm, param_distributions=p_grid,
                                           n_iter=n_iter_search, cv=inner_cv)
            clf = make_pipeline(StandardScaler(),Rnd_Srch)
    elif clf_name == 'linear_svm_scaled':
        svm = SVC(kernel='linear')
        # parameters for grid search
        if inner_cv=='None':
            clf=svm
        else:
            p_grid = {}
            p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
            # classifier
            n_iter_search=10
            Rnd_Srch = RandomizedSearchCV(svm, param_distributions=p_grid,
                                           n_iter=n_iter_search, cv=inner_cv)
            clf = make_pipeline(StandardScaler(),Rnd_Srch)
    elif clf_name == 'RF':
        if inner_cv==None:
            clf=RF()

        else:
            clf_init=RF()
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 50)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth}
            n_iter_search=10

            Rnd_Srch = RandomizedSearchCV(clf_init, param_distributions=random_grid,
                                           n_iter=n_iter_search, cv=inner_cv,iid=True)
            clf=Rnd_Srch
            #clf = make_pipeline(StandardScaler(),Rnd_Srch)

    return clf
#def run_features_selection(method='None',inner_cv=None,outer_cv=None):

def get_column(List, i):
    return [np.asarray(row[:,i]) for row in List]

def run_classification(clf_name=None,
                        X=[],
                        y=[],
                        groups=None,
                        inner_cv=None,
                        outer_cv=None,
                        mode='mf',
                        stat=None,
                        nb_permutations=100,
                        n_jobs=-1):
    """ Run classification procedure in multi or single feature

    Parameters
    ----------

    *clf: classifier name
    X=Feature matrice, should be n_samples x n_features
    y=Labels vector
    groups:
    stat : if true run permutations test
    nb_permutations=number of permutations to run if stat is True
    mode: string/by default = 'mf'; could be
        - 'mf' for multifeature classificatrun_classificationion without feature selection
        - 'mf_fs' for multifeature classification with feature selection
        - 'sf' for single feature classification
        - 'mf_sg_elect' for multifeature single electrode
    """
    if isinstance(X, list):
        n_samples,n_features=X[0].shape

        n_electrodes=len(X)
    else:
        n_samples,n_features=X.shape

    if mode=='sf':

        test_scores=[]
        permutation_scores=[]
        pvalues=[]

        for feat in range(n_features):
            clf= get_classifier(clf_name,inner_cv)
            if stat:

                print('\n for feature num: \n' , feat)
                test_score, permutation_score, pvalue = permutation_test_score(
                                                        clf, X[:,feat].reshape(-1,1), y,
                                                        scoring="accuracy",
                                                        cv=outer_cv,
                                                        groups=groups,
                                                        n_permutations=nb_permutations,
                                                        n_jobs=n_jobs)

                print("Test accuracy = %0.4f "%(test_score))
                test_scores.append(test_score)
                permutation_scores.append(permutation_score)
                pvalues.append(pvalue)



            else:
                print('Running classification without permutations')
                print('\n for feature num: \n' , feat+1)
                output = cross_validate(clf,
                                        X = X[:,feat].reshape(-1,1),
                                        y = y,
                                        cv = outer_cv,
                                        return_train_score = True,
                                        scoring='accuracy',
                                        n_jobs = n_jobs)
                                        #fit_params=fit_params,


                test_scores.append(output['test_score'].mean())
    if mode=='mf_sg_elect':
        test_scores=[]
        permutation_scores=[]
        pvalues=[]

        for elect in range(n_electrodes):
            clf= get_classifier(clf_name,inner_cv)
            if stat:

                print('\n for electrode:' , elect+1)
                print(X[elect].shape)
                test_score, permutation_score, pvalue = permutation_test_score(
                                                        clf, X[elect], y,
                                                        scoring="accuracy",
                                                        cv=outer_cv,
                                                        n_permutations=nb_permutations,
                                                        n_jobs=n_jobs)

                print("Test accuracy = %0.4f "%(test_score))
                test_scores.append(test_score)
                permutation_scores.append(permutation_score)
                pvalues.append(pvalue)

    elif mode=='mf': #(multi feature without feature selection)
        test_scores=[]
        permutation_scores=[]
        pvalues=[]
        clf= get_classifier(clf_name,inner_cv)

        if stat:
            print('Running multi feature classification with permutations')
            test_score, permutation_score, pvalue = permutation_test_score(
                                                    clf, X, y,
                                                    scoring="accuracy",
                                                    cv=outer_cv,
                                                    n_permutations=nb_permutations,
                                                    n_jobs=n_jobs)

            print("Test accuracy = %0.4f "%(test_score))
            test_scores.append(test_score)
            permutation_scores.append(permutation_score)
            pvalues.append(pvalue)

        else:
            print('Running Multifeature classification without permutations')

            output = cross_validate(clf,
                                    X = X,
                                    y = y,
                                    cv = outer_cv,
                                    return_train_score = True,
                                    scoring='accuracy',
                                    n_jobs = n_jobs)



            #print("Test accuracy = %0.4f +- %0.4f"%(output['test_accuracy'].mean(), output['test_accuracy'].std()))
            test_scores.append(output['test_score'].mean())

    # elif mode=='feature_selection':
    #     if method=='RFECV':


    return test_scores,permutation_scores,pvalues


def ExecuteRFECV(samples,y,featureNames,clusters,clusterNames,clf,kFolds,
                 nSplits,standardization,removedInfo,permutation,nPermutation,
                 currentDateTime,resultDir,debug,verbose):


    rfecv=RFECV(estimator=clf,
                cv=StratifiedKFold(kFolds),
                scoring='accuracy',
                n_jobs=-1)

    # Create empty Pandas dataframe
    cvResults           = pandas.DataFrame()
    decodingAccuracy    = pandas.DataFrame()
    permResults         = pandas.DataFrame()
    avg_perm_DA = []
    # Execute feature selection for nbOfSplit times
    for it in list(range(nSplits)) :
        # Randomly create stratified train and test partitions (1/3 - 2/3)
        xTrain,xTest,yTrain,yTest = tts(samples,y['Cluster'],
                                        test_size=0.33,
                                        stratify=y['Cluster'])
        # Data z-score standardization
        xTrainSet,zPrm = Standardize(xTrain,yTrain,standardization,debug)

        # "accuracy" is proportional to the number of correct classifications
        if verbose:
            print('  Fiting for split #{}'.format(it))
        rfecv.fit(xTrainSet,yTrain)

        # Append the dataframe with the new cross-validation results.
        cvResults['cv_Scores_'+str(it)]          = rfecv.grid_scores_
        cvResults['cv_Features_Rank_'+str(it)]   = rfecv.ranking_

        if debug:
            print('cvResults for it %d' % it)
            print(cvResults)

        # Plot number of features VS. cross-validation scores
        fig_cv = plt.figure(dpi=300)
        plt.subplot(211)
        plt.title('Best performance = %.2f with %d features' % \
                  (max(rfecv.grid_scores_), rfecv.n_features_))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score %")
        plt.plot(range(len(rfecv.grid_scores_)), rfecv.grid_scores_)

        # subplot selected features
        plt.subplot(212)
        plt.title('Features selection')
        plt.xlabel("Features")
        plt.xticks(range(len(rfecv.grid_scores_)),
                   featureNames,
                   rotation='vertical')
        plt.ylabel("Selection")
        plt.scatter(range(len(rfecv.grid_scores_)), rfecv.support_)
        plt.grid()
        plt.tight_layout()
        savedPlotName = resultDir+'RFECV'+'_CV_DA_'+clusters+'_'+str(it+1)+ \
                        '_'+str(nSplits)+'.png'
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig_cv)

        if verbose:
            print('\tComplete')

# ********************************** TEST *************************************
        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(xTest,zPrm)

        if verbose:
            print('  Testing')
        # use score() function to calculate DAs
        if debug:
            print('scores'+str(it))
            print(rfecv.score(xTestSet,yTest))
        decodingAccuracy['test_DA_'+str(it)] = [rfecv.score(xTestSet,yTest)]

        # plot confusion matrix
        y_pred = rfecv.predict(xTestSet)
        cm = confusion_matrix(yTest, y_pred)
        fig_CM = plt.figure(dpi=300)
        plot_confusion_matrix(cm, clusterNames, normalize=True, precision=2)
        savedPlotName = resultDir+'RFECV'+'_'+clusters+'_ConfusionMatrix_'+ \
                        str(it+1)+'_'+str(nSplits)+'.png'
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig_CM)

        if it == nSplits-1:
            print('\nTest Decoding accuracy')
            decodingAccuracy['test_Avg_DA']=decodingAccuracy.iloc[0][:].mean()
            for i in list(range(len(decodingAccuracy.iloc[0]))):
                print('\t'+str(decodingAccuracy.iloc[0].index[i])+'\t'+ \
                      str(decodingAccuracy.iloc[0][i]))

            #formating test results to save in excel file
            fTest = []
            for i in range(len(list(decodingAccuracy))-1):
                fTest.append(decodingAccuracy.iloc[0][i])

            testDA =pandas.DataFrame()
            testDA['test_DA_per_epoch'] = fTest
            tmp = pandas.DataFrame(data=[np.mean(testDA['test_DA_per_epoch'])],
                                         columns=['avg_test_DA'])

            testDA = pandas.concat([testDA,tmp],axis=1)
            print('\tComplete\n')


# ****************************** Permutation **********************************
        if permutation:
            if verbose:
                print('  Permutting')
            # Create subset based on selected best features
            xTrain_rfecv = rfecv.transform(xTrainSet)
            xTest_rfecv = rfecv.transform(xTestSet)
            permResults['permutation_DA_'+str(it)] = Permute(clusters,
                                                             xTrain_rfecv,
                                                             xTest_rfecv,
                                                             yTrain, yTest,
                                                             nPermutation,
                                                             debug_flag=0)
            avg_perm_DA.append(np.mean(permResults['permutation_DA_'+str(it)]))

#            savedHistName = resultDir+'/Permutation_hist_'+str(it)+'.png'
#            PlotPermHist(permResults,testDA.iloc[0][1],
#                         currentDateTime,savedHistName)
    if permutation:
        # compute permutation DA average and keep results in a dataframe
        epochedPermDA = ComputePermutationAvgDA(avg_perm_DA)

        print('Average permutation DA per train epoch')
        for i in epochedPermDA['Avg_Permutation_DA_per_epoch']:
            print('\t'+str(i))

        print('\nAverage permutation DA : {}'.format(
                            epochedPermDA['Global_Permutation_DA'][0]))

        savedHistName = resultDir+'Average_Permutation_hist.png'
        PlotPermHist(permResults,
                     testDA.iloc[0][1],
                     currentDateTime,
                     savedHistName)
        # formating permutation results to save in excel file
        permResults = pandas.concat([permResults,epochedPermDA], axis=1)


# ************************ Select best of best features ***********************
    ranks = cvResults.iloc[:,1::2]
    if debug:
        print(ranks)

    bestFeatures = pandas.DataFrame()
    bestFeatures = ranks[(ranks == 1).all(1)].index.tolist()
    print('\nBest features :')
    tmp = []
    for i in bestFeatures:
        tmp.append(featureNames[i])
        print('\t'+featureNames[i])
    bestFeaturesNames = pandas.DataFrame(data=tmp,columns=['Best_Features'])


    # Calculate number of time every features is selected as best
    bestFeaturesHist = ranks[(ranks == 1)].sum(axis=1)
    bestFeaturesHist.rename('Best_Features_Hist')

    # Build structure of histogram data to save in excel
    hist = pandas.DataFrame(data=featureNames,columns=['Features_Name'])
    hist['Occurence_Best'] = bestFeaturesHist
    nbSubject = pandas.DataFrame(data=[len(samples)],
                                 columns=['Number_Of_Subjects'])
    nbFeature = pandas.DataFrame(data=[samples.shape[1]],
                                       columns=['Number_Of_Features'])
    dataSize = pandas.concat([nbSubject,nbFeature],axis=1)

    # Get the best test DA and corresponding training set of features
    bestDA = testDA['test_DA_per_epoch'].max()
    bestDAepoch = testDA['test_DA_per_epoch'].idxmax()
    colName = 'cv_Features_Rank_'+str(bestDAepoch)
    bTrainFeat = cvResults[colName][(cvResults[colName] == 1)].index.tolist()
    tmp = []
    tmp.append(bestDA)
    for i in bTrainFeat:
        tmp.append(featureNames[i])
    bTrainFeatName = pandas.DataFrame(data=tmp,
                                      columns=['Best_Train_Features_Set'])

    # Build results structure to be save in excel file
    excelResults = pandas.concat([cvResults,
                                  testDA,
                                  permResults,
                                  hist,
                                  bestFeaturesNames,
                                  removedInfo,
                                  dataSize,
                                  bTrainFeatName],axis=1)
#    excelResults.to_excel(resultDir+'results_RFECV_'+currentDateTime+'.xlsx',
#                          sheet_name=xlSheetName)

    return excelResults




def CreateFolderArchitecture(FS_method):
    if not os.path.exists("Results"):
        os.mkdir("Results")
    currentDateTime = str(dt.datetime.now().strftime("%d%b%Y_%H-%M-%S"))
    resultDir = "Results/" + currentDateTime + '_' + FS_method + '/'
    os.mkdir(resultDir)

    return currentDateTime,resultDir

def RemoveNaNs(X,method,debug_flag):
    if method == 'Feature':
        rmTag = 'Removed_Features'
        removedData = X.columns[X.isnull().any()].tolist()
        X.dropna(axis=1,how='any',inplace=True)
    elif method == 'Subject': # error : not removing enough subjects...
        rmTag = 'Removed_Subject'
        removedData = X.index[X.isnull().any()].tolist()
        X.dropna(axis=0,how='any',inplace=True)

    print('Removed items : {}'.format(removedData))

    # Get the tag of the removed items, either features or subjects
    tmp=[]
    for i in list(range(len(removedData))):
        tmp.append(removedData[i])
    removedInfo = pandas.DataFrame(data=tmp,columns=[rmTag])

    return X, removedInfo

def GetClasses(dataset, classID):
    if classID == 'Sexe' or classID == 'Groupe' :
        tmp=cp.deepcopy(dataset[classID])
        y = pandas.DataFrame(data=np.reshape(tmp,[dataset.shape[0],1]),
                             columns=['Cluster'])
    elif classID == 'Mix' :
        tmp = []
        for i in list(range(len(dataset))):
            if dataset['Groupe'].iloc[i]==1 and dataset['Sexe'].iloc[i]==1:
                tmp.append([0,'youngWoman'])
            elif dataset['Groupe'].iloc[i]==2 and dataset['Sexe'].iloc[i]==1:
                tmp.append([1,'oldWoman'])
            elif dataset['Groupe'].iloc[i]==1 and dataset['Sexe'].iloc[i]==2:
                tmp.append([2,'youngMan'])
            elif dataset['Groupe'].iloc[i]==2 and dataset['Sexe'].iloc[i]==2:
                tmp.append([3,'oldMan'])
        y = pandas.DataFrame(data=tmp,columns=['Cluster','Cluster_Name'])

    return y

def GetSamples(dataset):
    X = cp.deepcopy(dataset)
    # Remove class ID columns
    X.drop(['Groupe','Sexe'],1, inplace=True)

    return X
"""
   Balance class so there is as many old men and old women in dataset
"""
def BalanceClasses(x,y):
    import random as rd

    X = cp.deepcopy(x)
    Y = cp.deepcopy(y)
    #    oldMen = y[y['Cluster_Name']=='oldMan']
    #    oldWomen = y[y['Cluster_Name']=='oldWoman']
    d = len(Y[Y['Cluster_Name']=='oldWoman']) - \
    len(Y[Y['Cluster_Name']=='oldMan'])
    if d > 0:
        idx = rd.sample(set(Y[Y['Cluster_Name']=='oldWoman'].index),d)
    else:
        idx = rd.sample(set(Y[Y['Cluster_Name']=='oldMan'].index),np.abs(d))
    # Ne veux pas dropper correctement...
    print('Dropped subject indexes: {}'.format(idx))
    X.drop(X.index[idx],inplace=True)
    Y.drop(idx,inplace=True)

    return X, Y

def Standardize(dataset,classes,method,debug_flag):
#    from scipy.stats import zscore
    # Not adapted for 4 groups classification
    if method == 'Across_Group': # Standardize by group
#        print('Standardize data across group of subjects')
#        uniqueY = np.unique(classes)
#        xSet = pandas.DataFrame()
#        for i in uniqueY:
#            tmp = pandas.DataFrame(data=cp.deepcopy(dataset[list(classes==i)]),
#                                   columns = ['Classe_'+str(i)])
#            print(tmp)
#            xSet.append(tmp)
##            xSet  =[xSet, newSet]
#            print(xSet)
        # Copy data of the two classes separatly
        youngX = cp.deepcopy(dataset[list(classes==1)]) # class 1 data
        oldX   = cp.deepcopy(dataset[classes==2]) # class 2 data

        # Remove class ID columns
        youngX.drop(['Groupe','Sexe'],1, inplace=True)
        oldX.drop(['Groupe','Sexe'],1, inplace=True)
        if debug_flag:
            print("Young without 'Group' and 'Sexe' column")
            print(youngX)
            print("Old without 'Group' and 'Sexe' column")
            print(oldX)
            print("Classes")

        # separate standardization with z-score
        youngZX,zParm1 = ComputeZScore(youngX)
        oldZX, zParam2 = ComputeZScore(oldX)
        zParam = [zParm1, zParam2]
        if debug_flag:
            print("Young standardized without 'Group' and 'Sexe' column")
            print(youngZX)
            print("Old standardized without 'Group' and 'Sexe' column")
            print(oldZX)

        # Combine young and old into a single matrix
        X = youngZX.append(oldZX)
        if debug_flag:
            print("Combined standardized data without 'Group' and 'Sexe' column")
            print(youngZX)

    elif method == 'Across_Subjects': # Standardization across all subject
#        print('Standardization data across all subjects')
        # Copy all dataset
        X = cp.deepcopy(dataset)
        # Remove class ID columns
#        X.drop(['Groupe','Sexe'],1, inplace=True)
        if debug_flag:
            print("All data without 'Group' and 'Sexe' column")
            print(X)

        # group standardization with z-score
        X, zParam = ComputeZScore(X, debug_flag)

        if debug_flag:
            print("All standardized data without 'Group' and 'Sexe' column")
            print(X)
            print('Standardization parameters')
            print(zParam)
    else: # not standardized
#        print('No standardization')
        X = cp.deepcopy(dataset)
        # Remove class ID columns
        X.drop(['Groupe','Sexe'],1, inplace=True)

        zParam = []

    return X, zParam

"""
Compute z-score on features that have unit

Return :    z-score transformation of dataset
            z-score parameter per feature
"""
def ComputeZScore(dataset, debug_flag=0):

    mns = []
    sstd = []
    result = cp.deepcopy(dataset)

    for i in list(dataset):
        if (i != 'EffSommeil' and i != 'Qualite subj.sommeil' and
            i != 'PSQI' and i != 'Rapport delta-theta' and
            i != 'Rapport theta-alpha' and i != 'Rapport theta-sigma' and
            i != 'Rapport theta-beta' and i != 'Rapport alpha-sigma' and
            i != 'Rapport alpha-beta' and i != 'Rapport sigma-beta'):
            m = dataset[i].mean()
            s = dataset[i].std()
            result[i] = (dataset[i] - m) / s
            mns.append(m)
            sstd.append(s)
        else:
            mns.append('')
            sstd.append('')

    if debug_flag:
        print('mean = {}'.format(mns))
        print('end mean')
        print('std = {}'.format(sstd))
        print('end std')

    tmp1 = pandas.DataFrame(data=mns,columns=['mean'])
    tmp2 = pandas.DataFrame(data=sstd,columns=['std'])
    param = pandas.concat([tmp1,tmp2],axis=1)
    return result, param

"""
Apply z-score stanrdadization to 'a' using mean and std from 'p'
"""
def ApplyStandardization(a,p):
    res = cp.deepcopy(a)
    for s in list(range(a.shape[0])):
        for f in list(range(a.shape[1])):
            if (p['mean'][f] != '' and p['std'][f] != ''):
                res.iloc[s][f] = (a.iloc[s][f]-p['mean'][f])/p['std'][f]
    return res

def ComputeRatio(X):
    X['Rapport delta-theta'] = X['Delta abs'] / X['Thêta abs']
    X['Rapport theta-alpha'] = X['Thêta abs'] / X['Alpha abs']
    X['Rapport theta-sigma'] = X['Thêta abs'] / X['Sigma abs']
    X['Rapport theta-beta']  = X['Thêta abs'] / X['Bêta abs']
    X['Rapport alpha-sigma'] = X['Alpha abs'] / X['Sigma abs']
    X['Rapport alpha-beta']  = X['Alpha abs'] / X['Bêta abs']
    X['Rapport sigma-beta']  = X['Sigma abs'] / X['Bêta abs']
    return X

def Permute(clusters,xTrain,xTest,yTrain,yTest,N,standardization,debug_flag=0):

    DA = [] # permutation's decoding accuracy

#    # Create subset based on selected best features
#    xTrain_estim = estimator.transform(xTrain)
#    xTest_estim = estimator.transform(xTest)
    # Create a classifier trainned with permutted label
    # Create the estimator and RFE object with a cross-validated score.
    if clusters == 'Mix':
        from sklearn.svm import LinearSVC
        permClf = LinearSVC(dual=False,multi_class='ovr')
    else:
        from sklearn.svm import SVC
        permClf = SVC(kernel="linear",shrinking=False)

    print('Permutting')
    for permIt in list(range(1,N+1)):
        print('\rPermutation {} of {} \r'.format(permIt,N),flush=True)

        # randomly permutte label to create the 'luck' probability
        permutted_yTrain = np.random.permutation(yTrain)
        # Data z-score standardization
        xTrainSet,zPrm = Standardize(xTrain,permutted_yTrain,standardization,0)
        # train classifier with permutted label
        permClf = permClf.fit(xTrainSet,permutted_yTrain)


        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(xTest,zPrm)
#        # Generate the new subsets based on the selected features
#        X_train_sfs = permClf.transform(xTrainSet.as_matrix())
#        X_test_sfs = permClf.transform(xTestSet.as_matrix())

#        # Fit the estimator using the new feature subset
#        # and make a prediction on the test data
#        permClf.fit(X_train_sfs, permutted_yTrain)
        y_pred = permClf.predict(xTestSet)

        # Compute the accuracy of the test prediction
        acc = float((yTest == y_pred).sum()) / y_pred.shape[0]
        if debug_flag:
            print('\nPSet accuracy\t: %.2f %%' % (acc * 100), end='\r\r')
        DA.append(acc)


#        # test classifier
#        y_pred = permClf.predict(xTest)
#        # Compute the accuracy of the prediction
#        acc = float((yTest == y_pred).sum()) / y_pred.shape[0]
#        if debug_flag:
#            print('Permutation #%d set accuracy\t: %.2f %%' % \
#                  (permIt,(acc * 100)))
#        DA.append(acc)
    print('')
    return DA

def ComputePermutationAvgDA(avgDA):
    # Create a dataframe with the received avergae DA
    DA1=pandas.DataFrame(data=avgDA,columns=['Avg_Permutation_DA_per_epoch'])
    # Add column containing the computed average DA of all iteration
    DA2 = pandas.DataFrame(data=[np.mean(DA1['Avg_Permutation_DA_per_epoch'])],
                           columns=['Global_Permutation_DA'])
    DA = pandas.concat([DA1,DA2],axis=1)

    return DA

def SaveExcel(fileName,sheetName,data,debug_flag=0):
    print('\nWritting the excel file')
    # load existing excel workbook or create it
    if os.path.isfile(fileName):
        workbook = openpyxl.load_workbook(fileName,0)
    else:
        workbook = openpyxl.Workbook()

    # create a new worksheet
    workbook.create_sheet(sheetName)
    # convert dataframe into something suitable for an excel worksheet
    rows = dataframe_to_rows(data,index=False)

    # Write new results to excel
    if debug_flag:
        print('DAs to write :')

    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            # write scores and ranks
             workbook[sheetName].cell(row=r_idx, column=c_idx, value=value)

    # save excel with modifications
    workbook.save(fileName)


class StratifiedShuffleGroupSplit(BaseEstimator):
    def __init__(self, n_groups, n_iter=None):
        self.n_groups = n_groups
        self.n_iter = n_iter
        self.counter = 0
        self.labels_list = []
        self.n_each = None
        self.n_labs = None
        self.labels_list = None
        self.lpgos = None
        self.indexes = None

    def _init_atributes(self, y, groups):
        if len(y) != len(groups):
            raise Exception("Error: y and groups need to have the same length")
        if y is None:
            raise Exception("Error: y cannot be None")
        if groups is None:
            raise Exception("Error: this function requires a groups parameter")
        if self.labels_list is None:
            self.labels_list = list(set(y))
        if self.n_labs is None:
            self.n_labs = len(self.labels_list)
        assert (
            self.n_groups % self.n_labs == 0
        ), "Error: The number of groups to leave out must be a multiple of the number of classes"
        if self.n_each is None:
            self.n_each = int(self.n_groups / self.n_labs)
        if self.lpgos is None:
            lpgos, indexes = [], []
            for label in self.labels_list:
                index = np.where(y == label)[0]
                indexes.append(index)
                lpgos.append(LeavePGroupsOut(self.n_each))
            self.lpgos = lpgos
            self.indexes = np.array(indexes)

    def split(self, X, y, groups):
        self._init_atributes(y, groups)
        y = np.asarray(y)
        groups = np.asarray(groups)
        iterators = []
        for lpgo, index in zip(self.lpgos, self.indexes):
            iterators.append(lpgo.split(index, y[index], groups[index]))
        for ite in product(*iterators):
            if self.counter == self.n_iter:
                break
            self.counter += 1
            train_idx = np.concatenate(
                [index[it[0]] for it, index in zip(ite, self.indexes)]
            )
            test_idx = np.concatenate(
                [index[it[1]] for it, index in zip(ite, self.indexes)]
            )
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups):
        self._init_atributes(y, groups)
        if self.n_iter is not None:
            return self.n_iter
        groups = np.asarray(groups)
        n = 1
        for index, lpgo in zip(self.indexes, self.lpgos):
            n *= lpgo.get_n_splits(None, None, groups[index])
        return n
