import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from copy import copy
from scipy.io import savemat, loadmat

import h5py

from joblib import Parallel, delayed

# def fonction(parametre1, parametre2):
#     load_data()
#     compute_something()
#     save_something(parametre1, parametre2)

#numpy.random.binomial(n, p, size=None)¶



# Parallel(n_jobs=n_jobs)(
#     delayed(fonction)(parametre1, parametre2)
#         for parametre1, parametre2 in itertools.product(liste1, liste2))


# pour si tu veux recuperer l'output de ta fonction juste mettre variable = Parallel(... varibale sera une liste.
def petropy(x,n,tau,method = 'order', accu = 4):
    """
    % The petropy function calculates the permutation entropy of data series.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Permutation Entropy
    %
    % H = PETROPY(X,N,TAU,METHOD,ACCU) computes the permutation entropy H of
    % a scalar vector X, using permutation order N, time lags from TAU and
    % METHOD to treat equal values. The ACCU parameter describes the accuracy
    % of the values in X by the number of decimal places.
    %
    % x      - data vector (Mx1 or 1xM)
    % n      - permutation order
    % tau    - time lag scalar OR time lag vector (length = n-1)
    % method - method how to treat equal values
    %   'noise' - add small noise
    %   'same'  - allow same rank for equal values
    %   'order' - consider order of appearance (first occurence --> lower rank)
    % accu   - maximum number of decimal places in x
    %         (only used for method 'noise')
    %
    % References:
    %
    % Bandt, C.; Pompe, B. Permutation Entropy: A Natural Complexity
    % Measure for  Time Series. Phys. Rev. Lett. 88 (2002) 17, 174102
    %
    % Riedl, M.; Müller, A.; Wessel, N.: Practical considerations of
    % permutation entropy. The European Physical Journal Special Topics
    % 222 (2013) 2, 249–262
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % example:
    %
    % H = petropy([6,9,11,12,8,13,5],3,1,'order');
    % H =
    %       1.5219
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """


    M = len(x)
    equal = False

    if n*np.log10(n) > 15:
        raise ValueError('permutation dimension too high')

    if isinstance(tau, (int, float)):
        tau = np.array([tau])
    else:
        if len(tau) > 1 and len(tau) != n-1:
            raise ValueError('time lag vector has to have n-1 entries')

    if (n-1)*np.min(tau) >=M  or np.max(tau) >= M:
        raise ValueError('too few data points for desired dimension and lags')


    if method.lower() == 'noise':
        #disp('Method: add small noise')
        x = x + np.random.rand(M)*10**(-accu-1)
    elif method.lower() ==  'equal':
        #disp('Method: allow equal ranks')
        equal = True
    elif method.lower() ==  'order':
        #disp('Method: consider order of occurrence')
        pass
    else:
        raise ValueError('unknown method')

    if isinstance(x, list):
        x = np.array(x)


    if len(tau) > 1:
        tau = np.reshape(tau, (len(tau),1));
        tau = np.concatenate(([0], tau));
        tau.sort()
        # build n x (M-tau(n))-matrix from shifted values in x
        shift_mat = np.zeros((n,M-tau(n)));
        for ii in range(n):
            shift_mat[ii,:] = x[tau[ii]:(M-tau[n]+tau[ii])]

    #########################################################
    else:
        # vectorized
        tmp1 = np.arange(0, (n-1)*tau+1, tau)
        tmp2 = np.arange(0, (M-(n-1)*tau))
        shift_mat_ind = np.tile(np.reshape(tmp1, (len(tmp1),1)), (1, (M-(n-1)*tau[0]))) + \
                        np.tile(np.reshape(tmp2, (1, len(tmp2))), (n, 1))
        shift_mat = x[shift_mat_ind]

    if equal:
        # allow equal values the same index
        ind_mat = np.zeros(shift_mat.shape)
        for ii in range(ind_mat.shape[1]) :
            ind_mat[:,ii]= np.unique(shift_mat[:,ii],  return_inverse=True)

    else:
        # sort matrix along rows to build rank orders, equal values retain
        # order of appearance

        sort_ind_mat = np.argsort(shift_mat, axis=0)
        ind_mat = np.zeros(sort_ind_mat.shape)
        for ii in range(ind_mat.shape[1]):
            ind_mat[sort_ind_mat[:,ii],ii] = np.arange(n)

    # assign unique number to each pattern (base-n number system)

    ind_vec = np.dot(n**np.arange(n), ind_mat)-1

    # find first occurence of unique values in 'ind_vec' and use
    # difference to determine length of sequence of the same numbers; e.g.
    # sort_ind_vec = [21 21 11 19 11], unique_values = [11 19 21],
    # ia = [1 3 4]: 11 occurs on places #1 and #2, 19 on #3 and 21 on #4 and #5
    ind_vec.sort()
    bidon, ia = np.unique(ind_vec, True)

    permpat_num = np.diff(np.concatenate((ia, [len(ind_vec)])));
    permpat_num = permpat_num/np.sum(permpat_num)

    # compute permutation entropy
    return -np.sum(permpat_num*np.log2(permpat_num))



def computePSD(signal, pageDuration):
    fs = len(signal)/pageDuration
    try:
        f,p = welch(signal, fs=fs, window='hamming', nperseg=int(len(signal)/6), noverlap=0, nfft=None)
    except ValueError:
        print(len(signal), pageDuration, int(len(signal)/6), fs)
    return f,p

def computePowerBands(f, amp):

    M_px=np.empty([6])
    M_px[0]=np.mean(amp[(f>=2)*(f <= 4)]) #delta
    M_px[1]=np.mean(amp[(f>=5)*(f <= 7)]) #theta
    M_px[2]=np.mean(amp[(f>=8)*(f <= 13)]) #alpha
    M_px[3]=np.mean(amp[(f>=8)*(f <= 13)]) #sigma
    M_px[4]=np.mean(amp[(f>=13)*(f <= 30)]) #beta
    M_px[5]=np.mean(amp[(f>=30)*(f <= 50)]) #low_gamma
    #M_px[5]=np.mean(amp[(f>=60)*(f <= 90)])


    return M_px

def computeRelPowerBands(f, amp):
    totPow = np.sum(amp)
    return (np.sum(amp[(f>=0.5)*(f <= 4.5)])/totPow,
            np.sum(amp[(f>=4.5)*(f <= 8.5)])/totPow,
            np.sum(amp[(f>=8.5)*(f <= 11.5)])/totPow,
            np.sum(amp[(f>=11.5)*(f <= 15.5)])/totPow,
            np.sum(amp[(f>=15.5)*(f <= 32.5)])/totPow )

def computeSpectralEntropy(f, amp):
    totPow = np.sum(amp)
    relPow = amp/totPow
    N = len(f)
    return -(1/np.log(N))*np.sum(relPow*np.log(relPow))

def computePowerbandsEpochs(data=[],epoch_length=30):
    print('Computing spectral powe density for all bands and electrodes..')
    nb_epochs=data.shape[2]
    nb_elect=data.shape[0]
    if nb_elect>nb_epochs:
        raise NameError('Data must be (elect x time x nb_epochs) shape ')
    else:
        Px_bd=np.empty([6,nb_epochs,nb_elect])
        for ep in range(nb_epochs):
            for elect in range(nb_elect):
                f,P=computePSD(data[elect,:,ep], epoch_length)
                bdpx=computePowerBands(f,P)
                Px_bd[:,ep,elect]=bdpx

    return Px_bd
