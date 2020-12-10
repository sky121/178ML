import numpy as np

from .base import classifier
from .base import regressor
from .utils import toIndex, fromIndex, to1ofK, from1ofK
from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from math import exp

################################################################################
## LINEAR CLASSIFY #############################################################
################################################################################


class linearClassify(classifier):
    """A simple linear classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
                  (1xN or CxN numpy array, where N=# features, C=# classes)

    Note: currently specialized to logistic loss
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for linearClassify object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        self.theta = np.array([])

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"

    #@property
    #def theta(self):
    #    """Get the linear coefficients"""
    #    return self._theta
    #def theta.setter(self,theta)
    #    self._theta = np.atleast_2d(


    def __repr__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


    def __str__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
                   len(self.theta), self.theta)
        return str_rep


## CORE METHODS ################################################################

    ### TODO: plot2D member function?  And/or "update line" & redraw? 
    ###   ALT: write f'n to take linear classifier & plot it
    ###   pass into gradient descent f'n as "updateFn" ?

    def predictSoft(self, X):
        """
        This method makes a "soft" linear classification predition on the data
        Uses a (multi)-logistic function to convert linear response to [0,1] confidence

        Parameters
        ----------
        X : M x N numpy array 
            M = number of testing instances; N = number of features.  
        """
        theta,X = twod(self.theta), arr(X)          # convert to numpy if needed
        resp = theta[:,0].T + X.dot(theta[:,1:].T)  # linear response (MxC)
        prob = np.exp(resp)
        if resp.shape[1] == 1:       # binary classification (C=1)
            prob /= prob + 1.0       # logistic transform (binary classification; C=1)
            prob = np.hstack( (1-prob,prob) )  # make a column for each class
        else:
            prob /= np.sum(prob,axis=1)   # normalize each row (for multi-class)

        return prob

    """
    Define "predict" here if desired (or just use predictSoft + argmax by default)
    """

    def train(self, X, Y, reg=0.0, initStep=1.0, stopTol=1e-4, stopIter=5000, plot=None):
        """
        Train the linear classifier.  
        """
        self.theta,X,Y = twod(self.theta), arr(X), arr(Y)   # convert to numpy arrays
        M,N = X.shape
        X1 = np.hstack((np.ones((M,1)),X))     # make data array with constant feature
        if Y.shape[0] != M:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        if self.theta.shape[1] != N+1:         # if self.theta is empty, initialize it!
            self.theta = np.random.randn(1,N+1)
        Y01 = toIndex(Y, self.classes)         # convert Y to "index" (binary: 0 vs 1)

        it   = 0
        done = False
        Jsur = []
        J01  = []
        while not done:
            step = (2.0 * initStep) / (2.0 + it)   # common 1/iter step size change

            for i in range(M):  # for each data point
                # compute linear response:
                respi = self.theta[:,0] + twod(X[i,:]).dot(self.theta[:,1:].T) 
                yhati = 1.0 if respi > 0 else 0.0   # convert to 0/1 prediction
                sigx  = np.exp(respi) / (1.0+np.exp(respi))
                gradi = -Y01[i]*(1-sigx)*twod(X1[i,:]) + (1-Y01[i])*sigx*twod(X1[i,:]) + reg*self.theta
                self.theta = self.theta - step * gradi

            # each pass, compute surrogate loss & error rates:
            Jsur.append( self.nll(X,Y) + reg*np.sum(self.theta**2) )
            J01.append( self.err(X,Y) )
            if plot is not None: plot(self,X,Y,Jsur,J01)
            #print Jsur

            # check stopping criteria:
            it += 1
            done = (it > stopIter) or ( (it>1) and (abs(Jsur[-1]-Jsur[-2])<stopTol) )


################################################################################
################################################################################
################################################################################
    def lossLogisticNLL(self, X,Y, reg=0.0):
        M,N = X.shape
        P = self.predictSoft(X)
        J = - np.sum( np.log( P[range(M),Y[:]] ) )   # assumes Y=0...C-1
        Y = ml.to1ofK(Y,self.classes)
        DJ= NotImplemented ##- np.sum( P**Y
        return J,DJ
        

    def myTrain(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        
        def sigma(r):
            try:
                return 1/(1+exp(-r))
            except OverflowError:
                if r < 0:
                    return 0.01
                return 0.99
            
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                #ri    = self.theta[0] + self.theta[1] * X[i,0] + self.theta[2] * X[i,1]; # TODO: compute linear response r(x)
                ri = np.dot(XX[i],self.theta.T)
                gradi = XX[i] * (-YY[i] + sigma(ri));     # TODO: compute gradient of NLL loss
                self.theta -= stepsize * gradi;  # take a gradient step

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            ##  Jsur = - sum_i [ (log si) if yi==1 else (log(1-si)) ]
            
            Jsur_sum = 0
            for i in range(M):
                ri = np.dot(XX[i],self.theta.T)
                if ri > 0:
                    Jsur_sum += np.log(sigma(ri))
                else:
                    Jsur_sum += np.log(1-sigma(ri))
            
            
            Jsur = -Jsur_sum/M
            
            Jnll.append( Jsur ) # TODO evaluate the current NLL loss
            #display.clear_output(wait=True)
            #plt.subplot(1,2,1); plt.cla(); plt.plot(Jnll,'b-',J01,'r-'); # plot losses
            #if N==2: plt.subplot(1,2,2); plt.cla(); self.plotBoundary(X,Y); # & predictor if 2D
            #plt.pause(.01);                    # let OS draw the plot        

            ## For debugging: you may want to print current parameters & losses
            #print(self.theta, ' => ', Jnll[-1], ' / ', J01[-1] , Jsur, M, Jsur_sum)
            #print(self.theta, ' => ', Jnll[-1], ' / ', J01[-1], Jsur)
            #print("epoch",epoch)

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            Jnll_length = len(Jnll) 
            #print("Difference: ", abs(Jnll[Jnll_length-1] - Jnll[Jnll_length-2]), "\tstop Tol: ", stopTol)
            if epoch > stopEpochs or (Jnll_length > 1 and abs(Jnll[Jnll_length-1] - Jnll[Jnll_length-2]) < stopTol):
                done = True;   # or if Jnll not changing between epochs ( < stopTol )

#    def TODOtrain(self, X, Y, reg=0.0, 
#                    initStep=1.0, stopTol=1e-4, stopIter=5000, 
#                    loss=None,batchsize=1,
#                    plot=None):
#        """
#        Train the linear classifier.  
#        """
#        self.theta,X,Y = twod(self.theta), arr(X), arr(Y)   # convert to numpy arrays
#        M,N = X.shape
#        if Y.shape[0] != M:
#            raise ValueError("Y must have the same number of data (rows) as X")
#        self.classes = np.unique(Y)
#        if self.theta.shape[1] != N+1:         # if self.theta is empty, initialize it!
#            self.theta = np.random.randn(1,N+1)
#
#        it   = 0
#        done = False
#        Jsur = []
#        J01  = []
#        while not done:
#            step = (2.0 * initStep) / (2.0 + it)   # common 1/iter step size change
#            for i in range(M):  # for each data point TODO: batchsize
#                _,gradi = loss(self,Xbatch,Ybatch)
#                self.theta = self.theta - step * gradi
#
#            # each pass, compute surrogate loss & error rates:
#            Jsur.append( loss(self,X,Y) )
#            J01.append( self.err(X,Y) )
#            if plot is not None: plot(self,X,Y,Jsur,J01)
#
#            # check stopping criteria:
#            it += 1
#            done = (it > stopIter) or ( (it>1) and (abs(Jsur[-1]-Jsur[-2])<stopTol) )
#
#
################################################################################
################################################################################
################################################################################

