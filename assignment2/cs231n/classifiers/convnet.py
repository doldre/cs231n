import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyConvNet(object):
    """
    Architecture

    [conv-bn-relu-pool] * 2 - [affine-bn-relu] * 2 - [affine-softmax]
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=5e-4, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network

        Input:
            - input_dim: Tuple(C, H, W) giving size of input data
            - num_filters: Number of filters to use in convolutional layer
            - filter_size: Size of filters to use in the convolutionallayer
            - hidden_dim: Number of units to use in the fully-connected 
            hidden layer
            - num_classes: Number of scores to produce from the final 
            affine layer.
            - weight_scale: Scalar giving standard deviation for random 
            initialization of weights.
            - reg: Scalar giving L2 regularization strength
            - dtype: numpy datatype to use for computation.
            """

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(
            num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(
            num_filters, num_filters, filter_size, filter_size)
        self.params['b2'] = np.zeros(num_filters)
        self.params['W3'] = weight_scale * np.random.randn(
            num_filters * H / 4 * W / 4, hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, 
                                                           hidden_dim)
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['W5'] = weight_scale * np.random.randn(hidden_dim, 
                                                           num_classes)
        self.params['b5'] = np.zeros(num_classes)

        print 'Use batchnorm!!'
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)
        self.params['gamma2'] = np.ones(num_filters)
        self.params['beta2'] = np.zeros(num_filters)
        self.params['gamma3'] = np.ones(hidden_dim)
        self.params['beta3'] = np.zeros(hidden_dim)
        self.params['gamma4'] = np.ones(hidden_dim)
        self.params['beta4'] = np.zeros(hidden_dim)
        self.bn_params = [{'mode': 'train'} for i in xrange(4)]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        for bn_param in self.bn_params:
            bn_param[mode] = mode

        # forward pass
        scores = None
        a1, c1 = conv_bn_relu_pool_forward(
            X, W1, b1, conv_param, 
            self.params['gamma1'], self.params['beta1'], self.bn_params[0],
            pool_param)
        a2, c2 = conv_bn_relu_pool_forward(
            a1, W2, b2, conv_param,
            self.params['gamma2'], self.params['beta2'], self.bn_params[1],
            pool_param)
        a3, c3 = affine_bn_relu_forward(
            a2, W3, b3, self.params['gamma3'], self.params['beta3'],
            self.bn_params[2])
        a4, c4 = affine_bn_relu_forward(
            a3, W4, b4, self.params['gamma4'], self.params['beta4'],
            self.bn_params[3])
        scores, c5 = affine_forward(a4, W5, b5)
        
        if y is None:
            return scores
        
        # backward pass
        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        dx, grads['W5'], grads['b5'] = affine_backward(dx, c5)
        dx, grads['W4'], grads['b4'], grads['gamma4'], grads['beta4'] = (
            affine_bn_relu_backward(dx, c4))
        dx, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = (
            affine_bn_relu_backward(dx, c3))
        dx, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = (
            conv_bn_relu_pool_backward(dx, c2))
        dx, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = (
            conv_bn_relu_pool_backward(dx, c1))

        # relularization
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + 
                                  np.sum(W3 * W3) + np.sum(W4 * W4) +
                                  np.sum(W5 * W5))

        grads['W5'] += self.reg * W5
        grads['W4'] += self.reg * W4
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads
        





