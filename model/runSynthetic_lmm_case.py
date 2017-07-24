import scipy.optimize as opt
import time

import sys

sys.path.append('../')
import numpy as np
from utility.helpingMethods import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from BOLTLMM import BOLTLMM
from LMMCaseControlAscertainment import LMMCaseControl

def rescale(x):
    maxi = np.max(np.abs(x))
    if maxi == 0:
        return x
    return x/maxi

def roc(beta, beta_true):
    beta = beta.flatten()
    beta = abs(beta)
    beta = rescale(beta)
    beta_true[beta_true != 0] = 1
    beta_true = beta_true.flatten()
    fpr, tpr, f = roc_curve(beta_true, beta)
    fp_prc, tp_prc, f_prc=precision_recall_curve(beta_true,beta)
    roc_auc = auc(fpr, tpr)
    return roc_auc




def cv_train( X, Y, beta,s,c,regMin=1e-30, regMax=1.0, K=100 ,maxEigen=None ):
    BETAs = []
    time_start = time.time()
    time_diffs = []
    print "begin"
    clf=LMMCaseControl()
    betaM = np.zeros((X.shape[1], Y.shape[1]))
    K = np.dot(X, X.T)
    for i in range(Y.shape[1]):

        if i%5==0:
            print "step================>  ",i

        clf.fit(X=X,y=Y[:,i],K=K, Kva=None, Kve=None, mode='lmm')
        betaM[:, i] =clf.getBeta()

    l=roc( betaM ,beta)
    BETAs.append(np.abs(betaM))
    print l
    time_end = time.time()
    time_diff = [time_end - time_start]
    print '... finished in %.2fs' % (time_diff[0])

    time_diffs.append(time.time() - time_start)
    return BETAs, time_diffs

def run(seed, n, p, g,  k, sigX, sigY,we, model,d=0.05, simulate=False):
    fileHead='/home/miss-iris/Desktop/tree/'
    fileHead = fileHead + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_' +str(we)+'_'+ str(seed) + '_'
    print fileHead
    X=np.load(fileHead + 'X.npy')
    Y=np.load(fileHead + 'Y.npy')
    beta=np.load(fileHead + 'beta1.npy')

    beta_n,time=cv_train(X=X,Y=Y,beta=beta,s=sigX,c=sigY)
    print fileHead,"ok~"
    np.save(fileHead+'beta_lmm_case',beta_n)

def run2(cat):
    c=cat
    n = 1000
    p = 5000
    d = 0.05
    g = 10
    k = 50
    sigX = 0.001
    sigY = 1
    we = 0.05
    model='tree'
    simulate=False
    for seed in [0, 1, 2, 3, 4]:
        print seed, "----", c
        if c == '0':
                run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'n':
                for n in [ 800,2000,1500]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'p':
                for p in [ 1000,2000, 8000]:#
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'g':
            for g in [ 5, 20, 50]:
                run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'k':
            for k in [5, 10, 100]:
                run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 's':
                for sigX in [0.0001,0.0005, 0.002]:#
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'c':
            # if seed==4:
                for sigY in [0.1, 0.5, 5]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'we':
                for we in [0.001, 0.01, 0.1]:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
        if c == 'd':
            for d in [0.01,0.03, 0.1]:
                if seed==3:
                    pass
                else:
                    run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)



if __name__ == '__main__':
    cat='0'
    run2(cat)
    cat='n'
    run2(cat)
    cat='p'
    run2(cat)
    cat='s'
    run2(cat)
    cat='c'
    run2(cat)
    cat = 'g'
    run2(cat)
    cat = 'k'
    run2(cat)
    cat = 'd'
    run2(cat)
    cat = 'we'
    run2(cat)

