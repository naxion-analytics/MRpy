from numpy.linalg import svd
from numpy import diagflat, outer,sqrt
from numpy import sum

__all__ = ['PCA']


class PCA():
    def __init__(self, n_components=2, plot_option = "symmetric"):
        self.n_components = n_components
        self.plot_option = plot_option
    
    def fit(self, X, y=None):
        P = X/sum(X)
        r = sum(P,axis=1)
        c = sum(P,axis=0)
        
        self.Dr = diagflat(1./sqrt(r))
        self.Dc = diagflat(1./sqrt(c))

        S = Dr.dot( P - outer(r,c)).dot(Dc)
        self._fit(S)
        
        return self
      
    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform()
        return self
    
    def transform(self, X=None):
        if X: 
            self.fit_transform(X)
            return self
        
        if self.plot_option == "symmetric":
            self.X = self._principleCoords(self.Dr, self.u)
            self.Y = self._principleCoords(self.Dc, self.v)
            
        elif self.plot_option == "rowprinciple":
            self.X = self._principleCoords(self.Dr, self.u)
            self.Y = self._standardCoords(self.Dc, self.v)
            
        elif self.plot_option == "colprinciple":
            self.X = self._standardCoords(self.Dr, self.u)
            self.Y = self._principleCoords(self.Dc, self.v)
        
        return self
    
    def _fit(self,X):
        u, s,vt = svd(X)
        self.v = vt[:self.n_components,:].transpose()
        self.u = u[:,:self.n_components]
        self.sigma = diagflat( s[:self.n_components])
    
    def _principleCoords(self, D, mat):
        return D.dot(mat.dot(self.sigma))
    
    def _standardCoords(self, D, mat):
        return D.dot(mat)
