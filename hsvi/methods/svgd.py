from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

def euc_dist(tX, X):
    k1 = tf.reshape(tf.reduce_sum(tf.square(tX),axis=1),[-1,1])
    k2 = tf.tile(tf.reshape(tf.reduce_sum(tf.square(X),axis=1),[1,-1]),[tX.shape[0].value,1])
    k = k1+k2-2*tf.matmul(tX,tf.transpose(X))

    return k

def rbf_kernel(pairwise_dist, h=1.):
    
    return tf.exp(-1.*pairwise_dist/h)

def cos_kernel(X):
    x_norm = tf.reduce_sum(tf.square(X),axis=1)
    X /= tf.reshape(x_norm,[-1,1])
    return tf.matmul(X,tf.transpose(X))

def get_median(v):    
    v = tf.reshape(v, [-1])
    m = v.shape[0].value//2
    return tf.nn.top_k(v, m).values[m-1]


class SVGD:
    def __init__(self,h=None):       
        self.h = h

    def svgd_kernel(self, X, kernel_type='rbf'):

        if kernel_type == 'rbf':
            if len(X.shape) > 2:
                X = tf.reshape(X,[X.shape[0].value,-1])
            pdist = euc_dist(X,X)

            if self.h is None:
                if X.shape[0].value == 1:
                    h = 1.
                else:
                    h = get_median(pdist)  
                h = tf.sqrt(0.5 * h / tf.log(X.shape[0].value+1.))

            kxy = rbf_kernel(pdist,h)

            dx = tf.expand_dims(X,[1]) - tf.expand_dims(X,[0])
            dkxy = 2*tf.matmul(tf.expand_dims(kxy,[1]),dx)/h
            #print('check shape',dkxy.shape)

        elif kernel_type == 'cos':
            kxy = cos_kernel(X)
            dkxy = tf.gradients(kxy,X)

        return kxy, tf.squeeze(dkxy,axis=1)

    def gradients(self,X,dlnprob):
        N = X.shape[0].value
        hdim = dlnprob.shape
        kxy, dkxy = self.svgd_kernel(X)
        if len(hdim) > 2 :        
            dlnprob = tf.reshape(dlnprob,[dlnprob.shape[0].value,-1])
        grad = -(tf.matmul(kxy, dlnprob) + dkxy)/N
        if len(grad.shape) != len(hdim):
            grad = tf.reshape(grad,hdim)
        return grad

    def update(self,X,dlnprob,vars,niter=1000,optimizer=None,sess=None):

        sgrad = self.gradients(X,dlnprob)

        if optimizer is None:
            global_step = tf.Variable(0, trainable=False, name="global_step")
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                                    global_step,
                                                                    100, 0.9, staircase=True)
            #from the implementation of the author
            optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.,beta2=0.9)

        train = optimizer.apply_gradients([(sgrad,vars)],global_step=global_step)

        if sess is None:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 
        tf.global_variables_initializer().run()

        
        for _ in range(niter):
            #print('sgrad',sess.run(sgrad))
            sess.run(train)