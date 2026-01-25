import numpy as np
import time

class LogisticRegression():
    def __init__(self,lr:float = 0.01,c:float=0.1,steps:int=1000,tol:float=0.01,verbose=False):
        self.lr = lr
        self.c = c
        self.steps= steps
        self.tol = tol
        self.verbose = verbose
        self.is_trained = False
        self.eps = 1e-9

    def _init_weights(self,X) -> tuple:
        r,c = np.asarray(X).shape
        self.w = np.random.randn(c,1)*0.01
        self.b = 0.0
        return r,c
    
    def _sigmoid(self,z):
        assert isinstance(z,np.ndarray)
        return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
        )

    def fit(self,X,y):
        IN = time.time()
        assert isinstance(X,(list,np.ndarray)) and isinstance(y,(list,np.ndarray))
        assert np.asarray(X).shape[0] == len(y)

        r,c = self._init_weights(X)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1,1)
        prev_loss = float('inf')

        for idx in range(self.steps):

            ##forward pass
            a = X @ self.w + self.b
            p = self._sigmoid(a)

            ##loss calculation
            loss = -np.mean(y*np.log(p+self.eps) + (1-y)*np.log(1-p+self.eps))
            if self.c:
                loss += self.c * np.sum(self.w**2)

            if np.abs((prev_loss - loss)) < self.tol:
                print(f'EARLY STOPPING TRIGGERED, exited in {time.time()-IN:.4f} seconds')
                break

            if self.verbose and idx%100 == 0:
                print('Step: {step}, Loss: {loss}'.format(step=idx,loss=loss))
            
            ##gradients 
            dLda = p - y
            dLdw = (X.T @ dLda) / r
            if self.c:
                dLdw += 2*self.c*self.w

            dLdb = np.mean(dLda)

            ##backward pass
            self.w -= self.lr*dLdw
            self.b -= self.lr*dLdb

            prev_loss = loss
        
        self.is_trained = True
        return self
    
    def predict(self,X):
        assert isinstance(X,(list,np.ndarray))
        assert self.is_trained, "[MODEL NOT TRAINED, CALL .fit()]"
        X = np.asarray(X)
        return (self._sigmoid(X@self.w + self.b) > 0.5).astype(int)




