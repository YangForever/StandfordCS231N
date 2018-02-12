import numpy as np

def RandomSearch(lossfunc, Xtr, Ytr):
    bestLoss = float('inf')
    # random from normal distribution
    ydim = Xtr.shape[1]
    xdim = Ytr.shape[1]
    lossRecord = []
    while i in range(10):  # or 1000 10000
        W = np.random.randn(xdim, ydim) # *0.001
        loss = lossfunc(Xtr, Ytr, W)
        lossRecord.append(loss)
        if loss < bestLoss:
            bestLoss = loss
            bestW = W

    return lossRecord, bestLoss, bestW

def RandomLocalSearch(lossfunc, Xtr, Ytr):
    bestLoss = float('inf')
    # random from normal distribution
    ydim = Xtr.shape[1]
    xdim = Ytr.shape[1]
    lossRecord = []
    bestW = np.random.randn(xdim, ydim) * 0.001
    while i in range(10):  # or 1000 10000
        Wtry = bestW + np.random.randn(xdim, ydim) * 0.001
        loss = lossfunc(Xtr, Ytr, Wtry)
        lossRecord.append(loss)
        if loss < bestLoss:
            bestLoss = loss
            bestW = Wtry

    return lossRecord, bestLoss, bestW

def FollowGradient(lossfunc, Xtr, Ytr):
    bestLoss = float('inf')
    # random from normal distribution
    ydim = Xtr.shape[1]
    xdim = Ytr.shape[1]
    lossRecord = []
    W = np.random.randn(xdim, ydim) * 0.001
    original_loss = lossfunc(Xtr, Ytr, W)
    lossRecord.append(original_loss)
    step_size = 0.01
    while i in range(10):  # or 1000 10000
        grad = eval_numerical_gradient(lossfunc, Xtr, Ytr, W)
        W = W - step_size * grad
        loss = lossfunc(Xtr, Ytr, W)
        lossRecord.append(loss)

    return lossRecord, W

def eval_numerical_gradient(LossFunc, Xtr, Ytr, W):
    fx = LossFunc(Xtr, Ytr, W)
    h = 0.00001
    grad = np.zeros(W.shape)
    
    '''
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            old_value = W[(i,j)]
            W[(i,j)] = old_value + h
            fxh = LossFunc(Xtr, Ytr, W)
            W[(i,j)] = old_value
            grad[(i,j)] = (fxh - fx) / h
    '''
    # replaced by 
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = W[ix]
        W[ix] = old_value+h
        fxh = LossFunc(Xtr, Ytr, W)
        W[ix] = old_value
        grad[ix] = (fxh-fx) / h
        it.iternext()
    return grad

        






