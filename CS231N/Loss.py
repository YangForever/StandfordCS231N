import numpy as np

def SVMLoss(Xtr, Ytr, W):
    N = Xtr.shape[0]
    delta = 1.0
    scores = np.dot(Xtr, W.T)
    loss = (1 / N) * np.sum(np.maximum(0, scores - scores[Ytr] + delta))
    return loss

def softmax(Xtr, Weights, Ytr, Yhop):

    scores = np.dot(Xtr, Weights)

    loss = 0
    Fmatrix = []
    exp = np.exp(scores)
    #print (exp)
    for i in range(len(scores)):
        lossi = -scores[i][Ytr[i]] + np.log(np.sum(np.exp(scores[i])))
        loss += lossi
        # cal F
        softm = np.exp(scores[i]) / np.sum(exp[i])
        Fmatrix.append(softm)
    loss = (1.0/len(scores)) * loss
    Fmatrix = np.asarray(Fmatrix)
    #print('f: ', Fmatrix)
    #print('Yhop: ', Yhop)
    dw = 1e-5 * Weights + np.dot(Xtr.T, (Fmatrix - Yhop))/10000
    # print(Fmatrix)

    #print("self loss: ", loss)
    #print("self dw: ", dw)

    return loss, dw

    # print(dw)
    # newWeights = Weights - 0.9 * dw * Weights
    # return newWeights
