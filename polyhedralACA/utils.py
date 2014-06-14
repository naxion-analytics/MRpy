import numpy as np

def Crop( mat, cols=None):
    mask = np.where(cols == 0)
    return np.delete(mat.toarray(), mask, axis=1)

def isFullRank(M):
	### update a's as well
    u,s,vt = np.linalg.svd(M)
    dep_idx = [i for i,l in enumerate(s) if abs(l)<1e-12]
    return np.delete(M, dep_idx, axis=0)

def RemoveJs(X,bigX, params, pref):
    n = len(np.where(pref!=0)[0] )
    a = np.zeros(len(pref) )
    a[np.where((pref!=0))] = params[:n]
    pref[np.where(a==0)] = 0.
    return Crop(X, cols=pref), Crop(bigX, cols=pref), pref