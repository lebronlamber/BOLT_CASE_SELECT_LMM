# import scipy.optimize as opt
# import time
#
# import sys
#
# sys.path.append('../')
#
# from lrLMM.helpingMethods import *
# from models.Lasso import Lasso
# from models.SCAD import SCAD
# from models.MCP import MCP
# from models.ProximalGradientDescent import ProximalGradientDescent
# from models.GroupLasso import GFlasso
# from models.TreeLasso import TreeLasso
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
#
# def rescale(x):
#     maxi = np.max(np.abs(x))
#     if maxi == 0:
#         return x
#     return x/maxi
#
# def roc(beta, beta_true):
#     beta = beta.flatten()
#     beta = abs(beta)
#     beta = rescale(beta)
#     beta_true[beta_true != 0] = 1
#     beta_true = beta_true.flatten()
#     fpr, tpr, f = roc_curve(beta_true, beta)
#     fp_prc, tp_prc, f_prc=precision_recall_curve(beta_true,beta)
#     roc_auc = auc(fpr, tpr)
#     return roc_auc
#
#
# def runLasso(X, Y, mode,model,maxEigen,flag):
#     learningRate=1e-5
#     if model == 'lasso':
#         time_start = time.time()
#         model = Lasso(lam=5e7, lr=learningRate)  #3e7 beta4
#         #print "lasso"
#         model.fit(X, Y)
#         time_end = time.time()
#         time_diff = [time_end - time_start]
#         print '... finished in %.2fs' % (time_diff[0])
#         return model.getBeta()
#
#
# def train_nullmodel( y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm',
#                     p=1):
#     """
#     train random effects model:
#     min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
#
#     Input:
#     X: Snp matrix: n_s x n_f
#     y: phenotype:  n_s x 1
#     K: kinship matrix: n_s x n_s
#     mu: l1-penalty parameter
#     numintervals: number of intervals for delta linesearch
#     ldeltamin: minimal delta value (log-space)
#     ldeltamax: maximal delta value (log-space)
#     """
#     ldeltamin += scale
#     ldeltamax += scale
#
#     y = y - np.mean(y)
#
#     if S is None or U is None:
#         S, U = linalg.eigh(K)
#
#     Uy = scipy.dot(U.T, y)
#     nllgrid = scipy.ones(numintervals + 1) * scipy.inf
#     ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
#     for i in scipy.arange(numintervals + 1):
#         nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods
#
#     nllmin = nllgrid.min()
#     ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]
#
#     for i in scipy.arange(numintervals - 1) + 1:
#         if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
#             ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
#                                                           (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
#                                                           full_output=True)
#             if nllopt < nllmin:
#                 nllmin = nllopt
#                 ldeltaopt_glob = ldeltaopt
#
#
#     return S, U, ldeltaopt_glob
#
# def cv_train( X, Y, beta,regMin=1e-30, regMax=1.0, K=100 ,maxEigen=None ):
#     BETAs = []
#     time_start = time.time()
#     time_diffs = []
#     print "begin"
#     # [n_s, n_f] = X.shape
#     # K = np.dot(X, X.T)
#     # X0 = np.ones(len(Y)).reshape(len(Y), 1)
#     # S, U, ldelta0 = train_nullmodel(y=Y, K=K)
#     #
#     # delta0 = scipy.exp(ldelta0)
#     # Sdi = 1. / (S + delta0)
#     # Sdi_sqrt = scipy.sqrt(Sdi)
#     # SUX = scipy.dot(U.T, X)
#     # SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
#     # SUy = scipy.dot(U.T, Y)
#     # SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
#     # SUX0 = scipy.dot(U.T, X0)
#     # SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
#
#     SUX=X
#     SUy=Y
#
#     for clf in [ MCP(), SCAD()]:
#         time_start = time.time()
#         lam_=5e7  #2e7 beta4
#         learningRate=1e-6
#         clf.setLambda(lam_)
#         clf.setLearningRate(learningRate)
#         # clf.fit(X, Y)
#         clf.fit(SUX, SUy)
#         betaM = clf.getBeta()
#         BETAs.append(np.abs(betaM))
#         l=roc( betaM,beta)
#         time_end = time.time()
#         time_diff = [time_end - time_start]
#         print '... finished in %.2fs' % (time_diff[0])
#         print l
#
#     betaM = runLasso(X,Y,mode='linear',model='lasso', maxEigen=0, flag=False)
#     BETAs.append(np.abs(betaM))
#     l=roc( betaM ,beta)
#     print l
#
#     time_diffs.append(time.time() - time_start)
#     return BETAs, time_diffs
#
# def run(seed, n, p, g, d, k, sigX, sigY, we, model, simulate):
#     #1load
#     fileHead='/home/miss-iris/Desktop/tree/'
#
#     fileHead = fileHead + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
#         sigX) + '_' + str(sigY) + '_' + str(we) + '_' + str(seed) + '_'
#     print fileHead
#     X=np.load(fileHead + 'X.npy')
#     Y=np.load(fileHead + 'Y.npy')
#     beta=np.load(fileHead + 'beta1.npy')
#
#     beta_n,time=cv_train(X=X,Y=Y,beta=beta)
#     print fileHead,"ok~"
#     np.save(fileHead+'beta5',beta_n)
# def run2(cat):
#     c=cat
#     n = 1000  # 800
#     p = 5000  # 8000
#     d = 0.05  # 0.06  0.03
#     g = 10
#     k = 50
#     sigX = 0.001  # 0.002
#     sigY = 1
#     we = 0.05
#     model='tree'
#     simulate=False
#     for seed in [0, 1, 2, 3, 4]:
#         print seed, "----", c
#         if c == '0':
#             run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'n':
#             for n in [ 800, 1500, 2000]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'p':
#             for p in [1000, 2000, 8000]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'g':
#             for g in [ 5, 20, 50]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'k':
#             for k in [5, 10, 100]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 's':
#             for sigX in [0.0001, 0.0005, 0.002]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'c':
#             for sigY in [0.1, 0.5, 5]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'we':
#             for we in [0.001, 0.01, 0.1]:
#                 run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         if c == 'd':
#             for d in [0.01,0.03, 0.1]:  # [0.03,0.04,0.06]:
#                 if seed==3:
#                     pass
#                 else:
#                     run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#         # if c == 'd2':
#         #     for d in [0.01, 0.1, 0.5]:  # 0.005 all and   all seed 3 not good
#         #         run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#
#
# if __name__ == '__main__':
#     cat='0'
#     run2(cat)
#     cat='n'
#     run2(cat)
#     cat='p'
#     run2(cat)
#     cat='s'
#     run2(cat)
#     cat='c'
#     run2(cat)
#     cat = 'g'
#     run2(cat)
#     cat = 'k'
#     run2(cat)
#     cat = 'we'
#     run2(cat)
#     cat = 'd'
#     run2(cat)


# import scipy.optimize as opt
# import time
#
# import sys
#
# sys.path.append('../')
# import numpy as np
# from utility.helpingMethods import *
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from BOLTLMM import BOLTLMM
#
# def rescale(x):
#     maxi = np.max(np.abs(x))
#     if maxi == 0:
#         return x
#     return x/maxi
#
# def roc(beta, beta_true):
#     beta = beta.flatten()
#     beta = abs(beta)
#     beta = rescale(beta)
#     beta_true[beta_true != 0] = 1
#     beta_true = beta_true.flatten()
#     fpr, tpr, f = roc_curve(beta_true, beta)
#     fp_prc, tp_prc, f_prc=precision_recall_curve(beta_true,beta)
#     roc_auc = auc(fpr, tpr)
#     return roc_auc
#
#
#
#
# def cv_train( X, Y, beta,s,c,regMin=1e-30, regMax=1.0, K=100 ,maxEigen=None ):
#     BETAs = []
#     time_start = time.time()
#     time_diffs = []
#     print "begin"
#     clf=BOLTLMM()
#     for f2 in [0.5, 0.3, 0.1]:  # fix parameters
#         for p in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:  # fix parameters
#             betaM = np.zeros((X.shape[1], Y.shape[1]))
#             for i in range(Y.shape[1]):
#                 temp=clf.train(X,Y[:,i],f2,p)  #,s,c
#                 temp=temp.reshape(temp.shape[0],)
#                 betaM[:, i]=temp
#                 if i%5==0:
#                     print "step: ",i
#
#             l=roc( betaM ,beta)
#             BETAs.append(np.abs(betaM))
#             print l
#     time_end = time.time()
#     time_diff = [time_end - time_start]
#     print '... finished in %.2fs' % (time_diff[0])
#
#     time_diffs.append(time.time() - time_start)
#     return BETAs, time_diffs
#
# def run(seed, n, p, g,  k, sigX, sigY, g_num,we, model,d=0.05, simulate=False):
#     #1load
#     fileHead='/home/miss-iris/Desktop/Group-LMM-master/Group-LMM-master/result/synthetic/group/'
#
#     fileHead = fileHead + '_' + str(n) + '_' + str(p) + '_' + str(k) + '_' + str(g) + '_' + str(g_num) + '_' + str(sigX) + '_' + str(sigY) + '_'+ str(we)+'_'+ str(seed) + '_'
#     print fileHead
#     X=np.load(fileHead + 'X.npy')
#     Y=np.load(fileHead + 'Y.npy')
#     beta=np.load(fileHead + 'beta1.npy')
#
#     beta_n,time=cv_train(X=X,Y=Y,beta=beta,s=sigX,c=sigY)
#     print fileHead,"ok~"
#     np.save(fileHead+'beta4',beta_n)
#
# def run2(cat):
#     c=cat
#     n = 1000
#     p = 5000
#     k = 50
#     g = 5
#     we = 0.05
#     g_num = 3
#     sigX = 1e-4
#     sigY = 50
#     model=''
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         # if c=='0':
#         print "~~~~~~~~~", seed
#         run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     print "--------------------------- 0"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         # if c=='we':
#         print "~~~~~~~~~", seed
#         for we in [0.01, 0.1]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     we = 0.05
#     print "--------------------------- we"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         # if c == 'n':
#         print "~~~~~~~~~", seed
#         for n in [500, 2000]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#
#     n = 1000
#     print "--------------------------- n"
#
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 'p':
#         for p in [2000, 10000]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     p = 5000
#     print "--------------------------- p"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 'k':
#         for k in [20, 100]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     k = 50
#     print "--------------------------- k"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 'g':
#         for g in [2, 10]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     g = 5
#     print "--------------------------- g"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 'gn':
#         for g_num in [2, 5]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     g_num = 3
#     print "--------------------------- gn"
#
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 's':
#         for sigX in [2e-5, 5e-4]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     sigX = 1e-4
#     print "--------------------------- s"
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         print "~~~~~~~~~", seed
#         # if c == 'c':
#         for sigY in [10, 100]:
#             run(seed=seed, n=n, p=p, g=g, k=k, sigX=sigX, sigY=sigY, model=model, g_num=g_num, we=we)
#     sigY = 50
#     print "--------------------------- c"
#
#
#
#
#
#     # if c == 'd2':
#         #     for d in [0.01, 0.1, 0.5]:  # 0.005 all and   all seed 3 not good
#         #         run(seed=seed, n=n, p=p, g=g, d=d, k=k, sigX=sigX, sigY=sigY, we=we, model=model, simulate=simulate)
#
#
# if __name__ == '__main__':
#     cat='0'
#     run2(cat)




import scipy.optimize as opt
import time

import sys

sys.path.append('../')
import numpy as np
from utility.helpingMethods import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from BOLTLMM import BOLTLMM

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
    clf=BOLTLMM()
    # for f2 in [0.5, 0.3, 0.1]:  # fix parameters
    #     for p in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:  # fix parameters
    f2=0.3
    p=0.1
    betaM = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        temp=clf.train(X,Y[:,i],f2,p)
        temp=temp.reshape(temp.shape[0],)
        betaM[:, i]=temp
        if i%5==0:
            print "step================>  ",i

    l=roc( betaM ,beta)
    BETAs.append(np.abs(betaM))
    print l
    time_end = time.time()
    time_diff = [time_end - time_start]
    print '... finished in %.2fs' % (time_diff[0])

    time_diffs.append(time.time() - time_start)
    return BETAs, time_diffs

def run(seed, n, p, g,  k, sigX, sigY,we, model,d=0.05, simulate=False):
    fileHead='/home/miss-iris/Desktop/Group-LMM-master/Group-LMM-master/result/synthetic/group/'
    fileHead='/home/miss-iris/Desktop/tree/'
    fileHead = fileHead + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_' +str(we)+'_'+ str(seed) + '_'
    print fileHead
    X=np.load(fileHead + 'X.npy')
    Y=np.load(fileHead + 'Y.npy')
    beta=np.load(fileHead + 'beta1.npy')

    beta_n,time=cv_train(X=X,Y=Y,beta=beta,s=sigX,c=sigY)
    print fileHead,"ok~"
    np.save(fileHead+'beta_bolt_lmm',beta_n)

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
                for n in [ 800,1500,2000]:
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
    cat = 'we'
    run2(cat)
    cat = 'd'
    run2(cat)


