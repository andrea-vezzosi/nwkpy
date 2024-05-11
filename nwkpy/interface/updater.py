# Broyden updater
import numpy as np
import scipy.linalg as linalg

def bound(x,lb,ub):
    """
    Bound real number inside the range [lb,ub]

    Input
    ----------
    x: float, number to be bound
    lb: float, lower bound 
    ub: float, upper bound

    Output
    ----------
    xbound: float, number inside the range

    """
    xbound = np.max((lb,np.min((ub,x))))

    return xbound

def normalize(x):
    norm = np.linalg.norm(x,axis=0)
    x = x / norm
    return x

class Updater:
    pass

class Broyden(Updater):
    """
    Broyden second method (inverse jacobian update).

    Reference articles:

    V. Eyert,
    A Comparative Study on Methods for Convergence Acceleration of Iterative Vector Sequences,
    Journal of Computational Physics,
    Volume 124, Issue 2,
    1996,
    Pages 271-285,
    ISSN 0021-9991,
    https://doi.org/10.1006/jcph.1996.0059.
    (https://www.sciencedirect.com/science/article/pii/S0021999196900595)

    Sun, L. & Yang, W. & Xiang, C. & Yu, Z. & Tian, L. (2005). 
    Broyden method for the self-consistent solution of Schrödinger and Poisson equations. 
    ASICON 2005: 2005 6th International Conference on ASIC, Proceedings. 2. 987-990. 

    """
    def __init__(self, N, M=8, beta=0.35, w0=0.01, use_wm=True):
        
        # target array dimension
        self.N = N
        
        # simple mixing parameter
        self.beta = beta
        
        # number of histories
        self.M = M
        
        # most recent history index
        self.n = -1
        
        # "black box" outer iteration
        self.m = 1
        
        # residual vector
        self.f = np.zeros((N,1))
        
        # difference vector
        self.dx = np.zeros((N,M))
        
        # residual difference vector
        self.df = np.zeros((N,M))  

        # weights initialization
        self.w = np.concatenate([np.array([w0]),np.ones(self.M)])

        # update weight for m>0 using Anderson rule wm = (f_m dot f_m)**(-1/2)
        self.use_wm = use_wm

    def update(self, xin, xout, reset=False):

        """
        Perform vector update using Broyden method.
        
        Input
        ----------
        xin: input vector of shape (N,1)
        xout: output vector from "black box" of shape (N,1)

        Output
        ----------
        xout: updated output vector of shape (N,1)

        """
        
        n = self.n
        m = self.m

        if n>-1:
            self.df[:,[np.min((n,self.M-1))]] = (xout - xin) - self.f
            #self.df = normalize(self.df)
            if self.use_wm:
                self.w[np.min((n,self.M-1))+1] = bound(x=np.linalg.norm(self.f)**(-1.),lb=1.,ub=1e12)
        self.f = xout - xin
        xout = xin + self.beta * self.f
        if reset:
            m=1
            n=-1
        k = np.min((m-1,self.M))
        
        if k>0:
            A = np.zeros((k,k))
            b = np.zeros((k,1))
            gamma = np.zeros((k,1))
            for i in range(k):
                for j in range(k):
                    A[i,j] = A[i,j] + self.w[i+1] * self.w[j+1] * np.dot( self.df[:,[i]].T , self.df[:,[j]] )
            for i in range(k):
                b[i] = b[i] + self.w[i+1] * np.dot( self.df[:,[i]].T , self.f )
                A[i,i] = A[i,i] + self.w[0]**2
            gamma = linalg.solve(A, b)
            for i in range(k):
                xout = xout - self.w[i+1] * gamma[i,0]*( self.dx[:,[i]] + self.beta*self.df[:,[i]] )
        if self.M>0:
            n = n+1
            if n>self.M-1:
                for i in range(self.M-1):
                    self.dx[:,i] = self.dx[:,i+1]
                    self.df[:,i] = self.df[:,i+1]
                    self.w[i+1] = self.w[i+2]
            self.dx[:,[np.min((n,self.M-1))]] = xout - xin
            #self.dx = normalize(self.dx)
        m = m+1
        self.n = n
        self.m = m
        
        return xout    
