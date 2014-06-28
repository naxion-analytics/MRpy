import numpy as np
from scipy import sparse
from scipy import optimize
from .utils import *

class polyhedralACA():

    def __init__(self,X,a):
        self.X = X
        self.a = a
        self.shape = X.shape

    @property
    def shape(self):
        return self.X.shape

    def _FindInternalPoint(self):
        """
        constrained optimization. find a point in the polyhedron defined by respondent responses and
        possible product designs
        """ 
        qr,pr = self.shape
    
        def length(x, sign = 1.0):   return sign * np.sum(x[pr:2*pr])
   
        cons = ({'type': 'eq', ### Xu = theta a
                 'fun': lambda x: np.array( self.X.dot(x[0:pr]) - x[2*pr]*self.a ) }, 
                {'type': 'ineq', ### y <= 1
                'fun': lambda x: np.ones(pr) - x[pr:2*pr]},
                {'type': 'ineq', ### u >= y
                'fun': lambda x: x[0:pr] - x[pr:2*pr]},
                {'type': 'ineq', ### theta >= 1
                'fun': lambda x: x[2*pr] -1},
                {'type': 'ineq', ### y >= 0
                'fun': lambda x: x[pr:2*pr]}
                )

        constrained_opt = optimize.minimize(length, 
                                    x0=np.random.random(2*pr+1),
                                    args=(-1.0,),
                                    method='SLSQP',
                                    constraints=cons,
                                    options={'disp':True})
        return constrained_opt

    def _FindCenter(self, u):
        """
        linesearch algorithm to find optimal alpha
        """
        qr,pr = self.shape
    
        u = params[:pr]
        U = sparse.dia_matrix( (u,0),shape=(pr,pr) )
        U2 = U.dot(U)
        U_inv = sparse.dia_matrix( (1./np.array(u),0), shape=(pr,pr))

        z  = np.linalg.inv( self.X.dot(U2.dot(X.T)) ).dot(self.a)
        d = u - U2.dot(self.X.T).dot(z)
    
        f = lambda x: -np.sum(np.log(x) )
        gradf = lambda x: - 1./x
    
        if np.linalg.norm(U_inv.dot(d) )<.25:
            alpha=[1.]
        else:
            alpha = optimize.line_search(f,gradf,u, d)
    
        return u+ alpha[0]*d


    def _FindLongAxis(self,center):
        """
        find the longest axis of the ellipse
        """
        qr,pr = self.shape

        u_asDiag =  [ 1./u**2 + 1./center[pr+(2*i+1)]**2    for i, u in enumerate(center[:pr])]
        U2 = sparse.dia_matrix( (u_asDiag, 0), shape=(pr,pr))

        z = np.linalg.inv(self.X.dot(X.T))
        normed_x = sparse.coo_matrix((self.X.T.dot(z) ).dot(self.X))
        M = (U2 - normed_x.dot(U2))

        spect, vecs = sparse.linalg.eigs(M)

        sorted_spec = [ np.real(eigval) if eigval>1e-15 else 10000. for eigval in spect  ]
        MIN_EIGVAL = np.min(sorted_spec)

        min_idx = np.where(sorted_spec == MIN_EIGVAL)[0]
        return np.array([np.real(v[0]) for v in vecs[:,min_idx]])






