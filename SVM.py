import numpy as np
from sklearn.datasets import load_iris    

class SVM:
    
    def __init__(self):
        self.w0 = np.random.normal(0,5,(3,1))
        
    def fit(self,X,y):
        
        m,n = X.shape
        X = np.hstack([np.ones([m,1]),X])
        Y = np.diag(y)
        Q  = np.eye(n  +1)
        Q[0][0] = 1e-8
        A = -np.dot(Y,X)
        c = np.zeros([n+1,1])
        b = -np.ones([m,1])
        print("demension : A{0}  Q{1}  Y{2}  c{3}  b{4}  X{5}".format(A.shape,Q.shape,Y.shape,c.shape,b.shape,X.shape))
        self.submodel = Quadratic_programming(Q,A,c,b)
        self.submodel.optimaizer(self.w0)   
        self.w =  self.submodel.w_opt
        
    def Classifier(self, z):

        return np.where(np.array(z) >= 0.0, 1, -1)

    def predict(self, X):
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])
        z = np.dot(X ,self.w0)
        pred = self.Classifier(z)
        return pred
    
    def margin(self,w0):
        
        return 1/np.sqrt(2*self.submodel.f(w0))
    
class Quadratic_programming:
    def __init__(self,Q,A,c,b):
        self.Q = Q
        self.A = A
        self.c = c
        self.b = b
        self.EPS = 1e-6
        self.flag =0
        self.n,self.m =A.shape
    def f(self,w):
        # demension  (1,2) * (2,2) - (1,2)*(2,1) = (1,1)
        return 0.5 * np.dot(np.dot(w.T,self.Q),w) - np.dot(self.c.T,w)
        
    def g(self,w):
        return np.dot(self.A,w) - self.b
    
    def gradient_f(self,w):
        return np.dot(self.Q,w) - self.c
    
    def gradient_g(self):
        return self.A.T
    
    def getInitial(self, w):
        for _ in range(1000):
            maxGIndex = np.where(self.g(w) == np.max(self.g(w)))[0][0]
            a = np.dot(self.A[maxGIndex] ,w).reshape(-1,1)
            h = np.abs(a-self.b[maxGIndex])
            k = np.dot(self.gradient_g()[:, maxGIndex].reshape(-1,1),h)
            w -= 1.1 * (k/ np.linalg.norm(self.A[maxGIndex]) ** 2)
            if self.KKT_g(w):
                break
        return w
    
    def parameter(self,H_index):
        if len(H_index) != 0:
            A_ = self.A[H_index,:]
            b_ = self.b[H_index]
            k,l = A_.shape
            M = np.dot(np.linalg.inv(np.vstack([np.hstack([A_,np.zeros([k,k])]),np.hstack([self.Q,A_.T])])),np.vstack([b_,self.c]))
            w_opt = M[0:self.m].reshape(-1,1)
            lambda_opt = np.zeros(self.m)
            lambda_opt[H_index] = M[self.n:len(M)]
        else:
            w_opt = np.dot(np.linalg.inv(self.Q),self.c)
            lambda_opt = np.zeros(self.m)
        return w_opt, lambda_opt
    
    def KKT_g(self,w):
        return (sum(self.g(w) < self.EPS) == len(self.g(w)))


    def optimaizer(self,w0):
        
        if not self.KKT_g(w0):
            for i in range(1000):
                w = np.random.rand(self.m,1)
                w=self.getInitial(w0)
                if self.KKT_g(w):
                    w0=w
                    H = np.where(np.abs(self.g(w0))<self.EPS)[0]
                    self.flag = 1
                    break
            if not self.flag:
                print("No initial value")
        
        
        for i in range(1,1000):
            w_opt,lambda_opt = self.parameter(H)
           
            if self.KKT_g(w_opt): #step4
                if sum(lambda_opt >= 0.0) == len(lambda_opt):
                    self.flag =1
                    
                else:
                    #min_index = np.where(lambda_opt == np.min(lambda_opt))
                    min_index  =np.argmin(lambda_opt)
                    H = np.sort(np.delete(H,min_index)) 
                    self.flag = 0
        
            else: #step3
                a= np.dot(self.A,(w_opt-w0))  #分母が0にならないように
                a[np.where(a == 0.0)]=self.EPS
                t =-self.g(w0)/a 
                if len(t[np.where(t > 0.0)]) == 0:
                    t = 0.0
                else:
                    t = np.min(t[np.where(t >0.0)])
                w0 +=  t*(w_opt-w0)
                H = np.where(np.abs(self.g(w0))<self.EPS)[0]
                self.flag = 0
                
            if self.flag:
                self.w_opt = w_opt
                
                break
        if not self.flag:
            print("No optimal solution w")
        
                

def main():
    iris = load_iris()
    X = iris.data[0:100,[1,3]]
    y = iris.target[0:100]
    y = np.where(y == 0,-1,1)
    model = SVM()
    model.fit(X,y)
    w = model.w

    margin =model.margin(model.w)
    print("margin :",margin)
    print("w:",w.reshape(1,-1))
    
    # plot
    marker=["o","o"]
    labels = ['setosa', 'versicolor']
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    for i, j in enumerate(np.unique(y)):
        plt.scatter(x=X[y == j, 0], y=X[y == j, 1],
                    marker=marker[i], label=labels[i])
    
    X1 = np.arange(x1_min, x1_max, 0.01)
    X2 = np.array(-w[1] / w[2] * X1 - w[0] / w[2])
    X2m= np.array(-w[1] / w[2] * X1 - (w[0] + 1.0) / w[2])
    X2p = np.array(-w[1] / w[2] * X1 - (w[0] - 1.0) / w[2])
    
    plt.plot(X1, X2, c="red")
    plt.plot(X1, X2m, c="black")
    plt.plot(X1, X2p, c="black")

    plt.xlabel("sepal width [cm]")  
    plt.ylabel("petal width [cm]")  
    plt.legend(loc="upper left")
    plt.show()
    
if __name__ == "__main__":
    main()
   
