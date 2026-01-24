import numpy as np
import time

class LinearRegression():
    def __init__(self,c:float=0.1,lr:float=0.01,steps:int=1000,tol:float=0.001,verbose:bool=False):
        self.lr = lr
        self.steps = steps
        self.c = c
        self.tol = tol
        self.verbose = verbose
        self.is_trained = False

    def _init_weights(self,X):
        r,c = np.asarray(X).shape
        self.w = np.random.randn(c,1)
        self.b = 0.0
        return r,c

    def fit(self,X,y):
        IN = time.time()
        assert isinstance(X,(list,np.ndarray)) and isinstance(y,(list,np.ndarray))
        assert np.asarray(X).shape[0] == len(y)

        ##initialize weights
        r,c = self._init_weights(X)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1,1)
        prev_loss = float('inf')
        ##training
        for idx in range(self.steps):
            
            #forward pass 
            y_hat = X @ self.w + self.b

            #loss calculation - MSE LOSS
            loss = np.mean((y_hat - y)**2)
            if self.c:
                loss += self.c*np.sum(self.w**2)

            if np.abs((prev_loss-loss)) < self.tol:
                print(f'EARLY STOPPING TRIGGERED, exited in {time.time()-IN:.4f} seconds')
                break
            
            if self.verbose and idx%100 == 0:
                print('Step: {step}, Loss: {loss}'.format(step=idx,loss=loss))

            dLdz = y_hat - y

            dLdw = (2/r)*(X.T@(dLdz))
            if self.c:
                dLdw+=(2/r)*self.c*self.w

            dLdb = 2*np.mean(dLdz)

            self.w -= self.lr*dLdw
            self.b -= self.lr*dLdb

            prev_loss = loss

        self.is_trained = True

        return self
    
    def predict(self,X):
        assert isinstance(X,(list,np.ndarray))
        assert self.is_trained, "[MODEL NOT TRAINED, CALL .fit()]"
        X = np.asarray(X)
        return X@self.w + self.b

        









