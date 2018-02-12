import numpy as np
import LoadCifar10
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        N, D = X.shape
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)
        scores = a1.dot(W2) + b2

        if y is None:
            return scores

        loss = 0
        exp = np.exp(scores)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        # Loss
        data_loss = -np.log(probs[range(N), y])
        data_loss = np.sum(data_loss) / N
        reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2*W2)
        loss = reg_loss + data_loss

        # grad
        grads = {}
        #print(probs.shape)
        dscores = probs
        dscores[range(N),y] -= 1
        dscores = dscores / N
        #print(dscores.shape)
        grads['W2'] = np.dot(a1.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        da1 = np.dot(dscores, W2.T)
        dz1 = da1
        dz1[ z1 <= 0 ] = 0
        grads['W1'] = np.dot(X.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        # L2 regression
        grads['W1'] += reg*W1
        grads['W2'] += reg*W2

        return loss, grads

    def Train(self, Xtr, ytr, Xval, yval,
              learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5,
              num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        N = Xtr.shape[0]
        iter_per_epoch = max(N / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        momentum = 0.9
        v_prev = 0.0
        v = 0.0
        
        v_prev2 = 0.0
        v2 = 0.0

    # Forward pass
        # Xtr : 10000x3072
        # init weights 3072x10
        # print(Xtr[1])
        # bias 1X10
        # bias = 0.01 * np.random.randn(1, CLASS)rain(X_train, y_train, X_val, y_val,
        for it in range(num_iters):
            indexs = np.random.choice(np.arange(N), batch_size)
            X_batch = Xtr[indexs]
            Y_batch = ytr[indexs]
            loss, grads = self.loss(X_batch, Y_batch, reg)

            loss_history.append(loss)
            #LOSS, DW = sample.softmax_loss_vectorized(weights, Xtr, Ytr)

    # backpropagation
            # weights = Loss.softmax_bp(Xtr, weights, Fmatrix, self.Yhop)
            
            # v_prev = v
            # v = momentum * v - learning_rate * grads['W1'] 
            v = momentum * v - learning_rate * grads['W1']
            self.params['W1'] += v # learning_rate * grads['W1'] # -momentum * v_prev + (1 + momentum) * v # learning_rate * grads['W1']

            self.params['b1'] -= learning_rate * grads['b1']

            # v_prev2 = v2
            # v2 = momentum * v2 - learning_rate * grads['W2']
            v2 = momentum * v2 - learning_rate * grads['W2']
            self.params['W2'] += v2 # learning_rate * grads['W2'] # -momentum * v_prev2 + (1 + momentum) * v2 # learning_rate * grads['W2']

            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 10 == 0:
                print('iterations: %d / %d, loss: %f ' %(it, num_iters, loss))
            if it % iter_per_epoch == 0:
                #print ('yb: ', Y_batch)
                #print('yv: ', yval)
                train_acc = (self.Predict(X_batch) == Y_batch).mean()
                #print(self.Predict(Xval))
                val_acc = (self.Predict(Xval) == yval).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate = learning_rate * learning_rate_decay
            
        # print (val_acc)
            #print("dw: ", dw)
        return loss_history, train_acc_history, val_acc_history

    def Predict(self, X):
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)
        scores = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        #print(y_pred)
        return y_pred

def LoadData():
    Xtrain, Ytrain, X_test, Y_test = LoadCifar10.load_CIFAR10()
    X_train = Xtrain[:9000,:]
    Y_train = Ytrain[:9000]
    X_val = Xtrain[9000:,:]
    Y_val = Ytrain[9000:]
    # print(Y_val)
    # print(len(Xtrain[0]))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def Plot(lh, tah, vah):
    plt.subplot(2,1,1)
    plt.plot(lh)
    plt.title('Loss history')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.subplot(2,1,2)
    plt.plot(tah, label='train')
    plt.plot(vah, label='val')
    plt.title('Trainacc and valacc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

Xtr, Ytr, Xval, Yval, Xte, Yte = LoadData()
Ytr = np.asarray(Ytr)
Yval = np.asarray(Yval)
nn = TwoLayerNet(3072, 50, 10)
lh, tah, vah = nn.Train(Xtr, Ytr, Xval, Yval, verbose=True, num_iters=3000, reg=0.5)
Plot(lh, tah, vah)
y_pred = nn.Predict(Xte)
test_acc = (y_pred == Yte).mean()
print (test_acc)
