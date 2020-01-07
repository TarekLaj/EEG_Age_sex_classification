import pandas
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as RandForestClf

from slpClass_toolbox import Standardize
from slpClass_toolbox import Permute
from slpClass_toolbox import ComputePermutationAvgDA
from slpClass_toolbox import PlotPermHist
from slpClass_toolbox import ApplyStandardization
from slpClass_toolbox import plot_confusion_matrix

def ExecuteRandForest(samples,y,featureNames,clusters,clusterNames,svc,kFolds,
                      nbOfSplit,nbrOfTrees,standardizationType,
                      removedData,permutation_flag,nbPermutation,
                      currentDateTime,resultDir,debug_flag,verbose):



    bestFeaturesHist = np.zeros([len(featureNames)])
    CvResult         = pandas.DataFrame()
    permResults      = pandas.DataFrame()
    tmpBest          = []
    DA               = []
    avg_perm_DA      = []

#**************************** TRAIN *******************************************
    for it in list(range(nbOfSplit)) :
        X_train,X_test,y_train,y_test = tts(samples,y['Cluster'],
                                            test_size=0.33,
                                            stratify=y['Cluster'])
        # Data z-score standardization
        xTrainSet,zPrm = Standardize(X_train,y_train,standardizationType,
                                     debug_flag)

        randForest = RandForestClf(n_estimators=nbrOfTrees,max_features=None,
                                   oob_score=True,n_jobs=-1)
        #**************************** SVM optimisation*************************
        params_dict={'C': scipy.stats.expon(scale=100),
                     'kernel': ['linear'],
                     'class_weight': ['balanced', None],
                     }

        n_iter_search = 20
        random_search = RandomizedSearchCV(svc,param_distributions=params_dict,
                                           n_iter=n_iter_search)

        random_search.fit(xTrainSet,y_train)
        clf = random_search.best_estimator_
        #**********************************************************************
        print('\nFitting for split #{}'.format(str(it)))
        randForest = randForest.fit(xTrainSet, y_train)

#***************************** TEST *******************************************
        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(X_test,zPrm)
        score = randForest.score(xTestSet, y_test)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        plot_step = 0.02  # fine step width for decision surface contours
        x_min, x_max = xTrainSet[:, 0].min() - 1, xTrainSet[:, 0].max() + 1
        y_min, y_max = y_train[:, 1].min() - 1, y_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        cmap = plt.cm.RdYlBu
        estimator_alpha = 1.0 / len(randForest.estimators_)
        for tree in randForest.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Plot the training points, these are clustered together and have a
        # black outline
        for i, c in zip(range(len(clusterNames)), "ryb"):
            idx = np.where(y == i)
            plt.scatter(xTrainSet[idx, 0], xTrainSet[idx, 1], c=c,
                        label=clusterNames[i],cmap=cmap)

        plt.suptitle("Classifiers on feature subsets of the Iris dataset")
        plt.axis("tight")

        plt.show()

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        Z = randForest.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]



        print('Best combination for fit nb %.1f (ACC: %.3f): %s' % \
              (it,randForest.k_score_, randForest.k_feature_idx_))
        fig1=plot_sfs(sfs1.get_metric_dict(), kind='std_err');
        savedPlotName = resultDir+'Decoding_accuracy_'+clusters+'_'+str(it)+\
                        '_'+str(nbOfSplit)+'.png'

#        tempBestfeat=np.asarray(sfs1.k_feature_idx_)
#        tmpBest.append(np.asarray(sfs1.k_feature_idx_))
        tmpBest.append(sfs1.k_feature_idx_)
        bestFeaturesHist[[tmpBest[-1]]]+=1
#        best_feat = n = np.hstack((best_feat,tempBestfeat)) if best_feat.size \
#                                                        else tempBestfeat
        fig1.set_dpi(300)
        plt.tight_layout()
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig1)

        fitRes = pandas.DataFrame.from_dict(sfs1.get_metric_dict()).T
        fitRes['avg_over_std'] = fitRes['avg_score'] / fitRes['std_dev']

        # plot mean / std
        plt.figure(dpi=300)
        plt.title('Moyenne sur ecart-type')
        plt.xlabel("nb attributs dans combinaison")
        plt.xticks(range(maxNbrFeaturesSFFS))
        plt.ylabel("Moyenne sur ecart-type")
        plt.plot(list(range(1,maxNbrFeaturesSFFS+1)),fitRes['avg_over_std'])
        figName = resultDir+'SFFS_'+clusters+'_bestSet_metric_'+ \
                                                    str(it)+'_'+str(nbOfSplit)
        plt.savefig(figName, bbox_inches='tight')

        # add metrics iteration identifier
        fitRes = fitRes.add_suffix('_'+str(it))

        CvResult = pandas.concat([CvResult,fitRes],axis=1)

#***************************** TEST *******************************************
        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(X_test,zPrm)

        # Generate the new subsets based on the selected features
        X_train_sfs = sfs1.transform(xTrainSet.as_matrix())
        X_test_sfs = sfs1.transform(xTestSet.as_matrix())

        # Fit the estimator using the new feature subset
        # and make a prediction on the test data
        clf.fit(X_train_sfs, y_train)
        y_pred = clf.predict(X_test_sfs)

        # Compute the accuracy of the test prediction
        acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
        print('Test set accuracy: %.2f %%' % (acc * 100))
        DA.append(acc)

        # plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_CM = plt.figure(dpi=300)
        plot_confusion_matrix(cm, clusterNames, normalize=True, precision=2)
        savedPlotName = resultDir+'SFFS_'+clusters+'_ConfusionMatrix_'+ \
                        str(it+1)+'_'+str(nbOfSplit)+'.png'
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig_CM)

#*************************** PERMUTATION **************************************
        if permutation_flag:
            permResults['permutation_DA_'+str(it)]=Permute(clusters,
                                                           xTrainSet,
                                                           xTestSet,
                                                           y_train, y_test,
                                                           nbPermutation,
                                                           standardizationType,
                                                           debug_flag=0)
            avg_perm_DA.append(np.mean(permResults['permutation_DA_'+
                                                   str(it)]))


            savedHistName = resultDir+'Permutation_hist_'+clusters+'_'+ \
                            str(it)+'.png'
#            PlotPermHist(permResults,DA[0][1],
#                         currentDateTime,savedHistName)
    dfDA = pandas.DataFrame(data=DA,columns=['DA_test'])
    CvResult = pandas.concat([CvResult,dfDA[:]],axis=1)
    CvResult = pandas.concat([CvResult,
                              pandas.DataFrame(data=[np.mean(DA)],
                            columns=['avg_DA'])],
                            axis=1)

#************************* STATISTICAL ASSESSMENT *****************************
    if permutation_flag:
        # compute permutation DA average and keep results in a dataframe
        print('\nAverage permutation DA')
        for i in list(range(len(avg_perm_DA))):
            print('\t'+str(avg_perm_DA[i]))

        savedHistName = resultDir+'Average_Permutation_hist_'+clusters+'.png'
        PlotPermHist(permResults,
                     CvResult['avg_DA'].iloc[0],
                     currentDateTime,
                     savedHistName)
            #formating permutation results to save in excel file
        permResults = pandas.concat([permResults,
                                     ComputePermutationAvgDA(avg_perm_DA)],
                                     axis=1)
        print('Mean permutation decoding accuracy : {}'.format(
                np.mean(permResults['Avg_Permutation_DA_per_epoch'])))
#    else : # binomial law
#        from scipy.stats import binom
#        q=0.001 # p value
#        n=300   # nombre d'observation (sujets)
#        p=0.5   # probablité d'avoir un essai correctement
#        chance_level= binom.isf (q, n, p) / n



    # Build structure of histogram data to save in excel
    hist = pandas.DataFrame(data=featureNames,columns=['Features_Name'])
    hist['Occurence_Best'] = bestFeaturesHist
    nbSubject = pandas.DataFrame(data=[len(samples)],
                                 columns=['Number_Of_Subjects'])

    #**************************************************************************
    # Search best set across every iteration best set
    best_Combination = tmpBest[np.argmax(DA)]

    # Compute average best combination size
    l=0
    for n in list(range(len(tmpBest))):
        l += len(tmpBest[n])
    avgBestCombSize = pandas.DataFrame(data=[np.ceil(l/len(tmpBest))],
                                       columns=['avgBestCombSize'])

#    subsetHist = GetSubsetOccurence(tmpBest)
#    PlotHist(subsetHist[1],'Subsets occurences',subsetHist[0],'Comb_Hist.png')
    #**************************************************************************
#************************** Build result structure ****************************
    tmp=[]
    tmp.append(np.max(DA))
    for i in best_Combination:
        tmp.append(featureNames[i])
        print('\t'+featureNames[i])
    bestFeatNames = pandas.DataFrame(data=tmp,columns=['Best_Features_Set'])

    #save training and permutation results in an excel file
    excelResults = pandas.concat([CvResult,
                                  permResults,
                                  hist,
                                  bestFeatNames,
                                  avgBestCombSize,
                                  removedData,
                                  nbSubject], axis=1)

    # Plot best combination custom metric (mean / std_dev)
    from slpClass_toolbox import PlotBestCombinationMetrics

    filteredData = excelResults.filter(regex=r'avg_over_std_', axis=1)
    metrics = pandas.DataFrame(data=filteredData)
    metrics.dropna(inplace=True)
    figName = resultDir+'SFFS_'+clusters+'_bestSet_metric_aggreg.png'

    PlotBestCombinationMetrics(metrics,figName)

    print('Mean Decoding accuracy :{}'.format(np.mean(DA)))

    # compute occurence of every subset in bestsets of every iteration
    from slpClass_toolbox import GetSubsetOccurence
    subsetHist = GetSubsetOccurence(tmpBest)
    excelResults = pandas.concat([excelResults, subsetHist], axis=1)
#    excelResults.to_excel(saveTo, sheet_name=xlSheetName)

    return excelResults
