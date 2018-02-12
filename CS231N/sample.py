import numpy as np

def softmax_loss_vectorized(W, X, y):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  f = X.dot(W)
  # f -= np.max(f, axis=1, keepdims=True) # max of every sample
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  p = np.exp(f)/sum_f

  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1

  print('f: ', p)
  print('Yhop: ', ind)

  dW = X.T.dot(p - ind)

  loss /= num_train
  # loss += 0.5 * reg * np.sum(W * W)
  # dW /= num_train
  dW += 1e-5*W
  print("self loss: ", loss)
  print("self dw: ", dW)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
