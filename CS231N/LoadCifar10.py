import pickle

filepath = 'data/cifar-10-batches-py'

def load_CIFAR10(): #filepath):
    Xtr, ytr = load_Train(filepath)
    #print('final')
    #print(len(ytr))
    #print(Xtr.shape)
    Xte, yte = load_Test(filepath)
    return Xtr, ytr, Xte, yte

def load_Train(filepath):
    Xtr = []
    ytr = []
    # for i in range(1, 6):
    datapath = filepath + '/data_batch_1' # + str(i)
    with open(datapath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    Xtr = dict[b'data']
    ytr = dict[b'labels']
    # print(len(ytr))
    return Xtr, ytr

def load_Test(filepath):
    Xte = []
    yte = []
    datapath = filepath + '/test_batch'
    with open(datapath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    Xte = dict[b'data']
    yte = dict[b'labels']
    return Xte, yte

Xtr, ytr, Xte, yte = load_CIFAR10()
# print(ytr[0:3])