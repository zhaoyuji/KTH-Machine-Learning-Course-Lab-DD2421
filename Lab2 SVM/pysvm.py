import numpy as np
import random as r
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import matplotlib.patches as mpatches
import time

warnings.filterwarnings('ignore')

class SVM():
    def __init__(self, data, t, kernel=['linear', None], 
          slack=True, C=1.0):
        self.data = data
        self.t = t
        self.N = data.shape[0] 
        self.C = C
        self.slack = slack
        self.kernel = kernel
        self.k = self.kernel_select()
        self.P = self.MatrixFormation()

        
    
    def kernel_select(self):
        if self.kernel[0] == 'linear':
            def LinearKernel(x, y):
                return np.dot(x, y)
            return LinearKernel
        
        elif self.kernel[0] == 'poly':
            def PolynomialKernel(x, y, p=self.kernel[1]):
                return np.power(np.dot(x, y)+1, p)
            return PolynomialKernel
        
        if self.kernel[0] == 'rbf':
            def RBFKernel(x, y, sigma=self.kernel[1]):
                return np.exp(-np.linalg.norm(x-y, 2)**2/(2*np.power(sigma,2)))

            return RBFKernel
    
    def defineK(self):
        N = self.N
        data = self.data
        
        K = np.zeros([N, N])
        
        for i in range(N):
            for j in range(N):
                if i <= j:
                    res = self.k(data[i], data[j])
                    K[i,j] = res
                    K[j,i] = res
        return K
    
    def MatrixFormation(self):
        
        N = self.N
        K = self.defineK()
        t = self.t
        
        P = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P[i][j] = t[i] * t[j] * K[i,j]
        self.P = P
        return P
        
    
    def ObjFunc(self,args):
        N, P = args
        res = lambda a: 0.5*np.sum([ [a[i] * a[j] * P[i,j] for i in range(N)] for j in range(N)]) - np.sum(a)
        return res
    
    
    def zerofun(self, a):
        return np.dot(self.t, a)
    
    
    def optimize(self):

        data = self.data
        t = self.t
        N = self.N
        P = self.P
        slack = self.slack
        C = self.C
        
        st = np.zeros(len(data))
    
        if slack == True:
            B = [(0, C) for b in range(N)]
        else:
            B = [(0, None) for b in range(N)]
        XC = {'type':'eq', 'fun':self.zerofun}
    
        args = (N, P)
        ret = minimize(fun=self.ObjFunc(args), x0=st, bounds=B, constraints=XC)
        if ret['success'] == True:
            alpha = ret['x']
        else:
            print("\n---------- No Solution ! ----------------\n")
            return None, None
        
            
        
        threshold = 1e-5
        
        falpha = np.array([i if i >= threshold else 0 for i in alpha])
        self.falpha = falpha
        
        sv = [(data[i], t[i], falpha[i]) for i in range(len(data)) if falpha[i] != 0]
        self.sv = sv
        
        return falpha, sv
        
    def CalB(self, sv):
        data = self.data
        t = self.t
        N = self.N
        
        calres = []
        for m in sv:
            s = m[0]
            res = np.sum([self.falpha[i]*t[i]*self.k(s,data[i]) for i in range(N) ])
            calres.append(res-m[1]) 
        return np.sum(calres)/len(sv)
#        idx = 0
#        for i in range(N):
#            if self.falpha[i] != 0:
#                idx = i
#                break
#        ans = 0
#        for i in range(len(data)):
#            ans += self.falpha[i]*t[i]*kernel(data[idx], data[i])
#        return ans - t[idx]


    def Indicator(self, td):
        sv = self.sv
        self.b = self.CalB(sv)
        return np.sum([m[2]*m[1]*self.k(td, m[0]) for m in sv]) - self.b
    
    def plotBoundary(self, classA, classB):

        xgrid = np.linspace(-3, 3)
        ygrid = np.linspace(-3, 3)
        
        
        #grid = np.array([[self.Indicator([x,y]) for x in xgrid] for y in ygrid])
        print("Calculating the values of indicator function...")
        time.sleep(0.5)
        grid = []
        for y in tqdm(ygrid):
            _tmp = []
            for x in xgrid:
                _tmp.append(self.Indicator([x,y]))
            grid.append(_tmp)
                
        grid = np.array(grid)
        
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
    
    
        blue_patch = mpatches.Patch(color='blue', label='ClassA')
        red_patch = mpatches.Patch(color='red', label='ClassB')
        black_patch = mpatches.Patch(color='black', label='Decision Boundry')
        plt.legend(handles=[blue_patch, red_patch, black_patch])
        
        plt.hold(True)
        plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
        plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
        plt.show()
    


if __name__ == '__main__':
    np.random.seed(100)
    classA = np.concatenate(
            (np.random.randn(10,2)*0.2+[1.5,0.5],
             np.random.randn(10,2)*0.2+[-1.5,0.5])) 
    classB = np.random.randn(20,2)*0.2+[0.0,-0.5]
    inputs = np.concatenate(( classA , classB )) 
    targets = np.concatenate (
            (np.ones(classA.shape[0]),
             -np.ones(classB.shape[0])))
    permute=list(range(len(inputs))) 
#    r.shuffle(permute) 
    data = inputs[permute, :]
    t = targets[permute]    
    
#    svm = SVM(data, t, kernel=['rbf', 1], 
#              slack=True, C=1.0)
#    falpha, sv = svm.optimize()

    svm = SVM(data, t, slack=True, C=1.0)
    falpha, sv = svm.optimize()
    svm.plotBoundary(classA, classB)

#    svm = SVM(data, t, slack=False)
#    falpha, sv = svm.optimize()
#    svm.plotBoundary(classA, classB)    
    
    
#    
#    def construct_dataset(classA, classB):
#        data = np.concatenate([classA, classB])
#        return data[:,:2], data[:,2]
#    
#    
#    import sklearn.datasets as dt
#    X, Y = dt.make_circles(100, factor=0.2, noise=0.1)
#    classC = []
#    for i in range(len(X)):
#        classC.append((X[i][0], X[i][1], 1 if Y[i] == 1 else -1))
#    
#    classA = [a for a in classC if (a)[2] == 1]
#    classB = [a for a in classC if (a)[2] == -1]
#    
#    
#    data, t = construct_dataset(classA, classB)
#    svm = SVM(data, t, slack=False)
#    falpha, sv = svm.optimize()
#    svm.plotBoundary(classA, classB) 
#        
#        
        
    
    
    
    
    
    
