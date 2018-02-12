import numpy as np
import LoadCifar10 

class NearestNeighbours(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X = X
        self.y = y
   
    def predict(self, X):
        num_test = X.shape[0]
        print (num_test)
        Ypred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            distances = np.sum(np.abs(self.X-X[i, :]), axis=1) 
            # default axis return a integer by adding all together
            # axis=0 adds each collum
            # axis=1 adds each row
            min_index = np.argmin(distances)
            Ypred[i] = self.y[min_index]
            #print ('finish: ' + str(i))
        return Ypred

    def predictKnh(self, X, nh):
        num_test = X.shape[0]
        print (num_test)
        Ypred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            print ('i:' + str(i))
            distances = np.sum(np.abs(self.X-X[i, :]), axis=1) 
            # default axis return a integer by adding all together
            # axis=0 adds each collum
            # axis=1 adds each row
            sortedIndex = distances.argsort()
            y_pred = {}
            for j in range(nh):
                min_index = sortedIndex[j]
                classval = self.y[min_index]
                # dic.get(key, default) if key then return key, otherwise return default
                y_pred[classval] = y_pred.get(classval, 0) + 1
            sortedclass = sorted(y_pred,key=y_pred.get,reverse=True)
            print (sortedclass)
            print (y_pred)
            Ypred[i] = sortedclass[0]
            print ('Ypred: ', Ypred[i])
            #print (sortedclass[0])
            #print ('finish: ' + str(i))
        return Ypred


if __name__ == '__main__':
    
    
    Xtr, Ytr, Xte, Yte = LoadCifar10.load_CIFAR10('data/cifar-10-batches-py')
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    # Test nn
    #nn = NearestNeighbours()
    #nn.train(Xtr_rows, Ytr)
    #Yte_pred = nn.predict(Xte_rows)
    #accRate = len(set(Yte_pred) & set(Yte)) / len(Xte)
    #print (accRate)

    # Test KNN
    nn = NearestNeighbours()
    nn.train(Xtr_rows, Ytr)
    Yte_pred = nn.predictKnh(Xte_rows[1:31], 10)
    print (Yte_pred)
    print (Yte[1:31])
    accRate = len(set(Yte_pred) & set(Yte[1:31])) / len(Xte[1:31])
    print (accRate)


