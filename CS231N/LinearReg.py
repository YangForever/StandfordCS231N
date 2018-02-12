import numpy as np
import Loss
import LoadCifar10
import sample

CLASS = 10

def LoadData():
    Xtrain, Ytrain, Xtest, Ytest = LoadCifar10.load_CIFAR10()
    return Xtrain, Ytrain, Xtest, Ytest

def genYhop(Ytr):
    Yhop = np.zeros((len(Ytr), CLASS), dtype=int)
    for i in range(len(Ytr)):
        Yhop[i][Ytr[i]] = 1
    return Yhop

class LinearReg(object):
    def __init__(self, Yhop):
        self.Yhop = Yhop

    def DataNormalisation(self, data):
        Xtr = data - np.mean(data, axis=0)
        Xtr /= np.std(Xtr, axis=0)
        return Xtr

    def Train(self, Xtr, Ytr):
    # Forward pass
        # Xtr : 10000x3072
        # init weights 3072x10
        weights = 0.001 * np.random.randn(Xtr.shape[1], CLASS)
        # print(Xtr[1])
        # bias 1X10
        # bias = 0.01 * np.random.randn(1, CLASS)
        for epoch in range(30):
            loss, dw = Loss.softmax(Xtr, weights, Ytr, self.Yhop)
            #LOSS, DW = sample.softmax_loss_vectorized(weights, Xtr, Ytr)
    # backpropagation
            # weights = Loss.softmax_bp(Xtr, weights, Fmatrix, self.Yhop)
            weights -= 1e-2*dw
            print("epoch: ", epoch)
            print("loss: ", loss)
            #print("dw: ", dw)
        return weights

'''
Xtr = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
weights = np.asarray([[0.1,0.2], [0.3,0.4],[0.5,0.6]])
Ytr = [0,0,1,1]
Yhop = genYhop(Ytr)
LR = LinearReg(Yhop)
LR.Train(Xtr, Ytr, weights)
'''


Xtr, Ytr, Xte, Yte = LoadData()
Yhop = genYhop(Ytr)
LR = LinearReg(Yhop)
Xtr_nor = LR.DataNormalisation(Xtr)
LR.Train(Xtr_nor, Ytr)
