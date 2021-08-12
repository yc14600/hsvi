from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import six


from utils.distributions import RandomVariable


class Hierarchy_SVI:
    def __init__(self,latent_vars={},data={},scale={},optimizer={},clipping={},vi_types={},\
                train_size=1000, coresets={}, extra_loss={},*args,**kwargs):
        
        print('start init hsvi')
        
        self.latent_vars = latent_vars
        self.data = data
        self.scopes = list(latent_vars.keys())
        self.scale = scale
        self.optimizer = optimizer
        self.clipping = clipping
        self.vi_types = vi_types
        self.train_size = train_size
        self.coresets = coresets  
        

        # to add any other loss 
        self.extra_loss = extra_loss

        # obtain variable list
        self.var_dict = {}               
        for scope in self.scopes:
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
                tmp.add(v)
            self.var_dict[scope] = list(tmp)
        
        self.losses, grads_and_vars = self.build_loss_and_gradients(self.var_dict)       
        self.grads = grads_and_vars # add for debug

        self.train = {}
        for scope in self.scopes:
            self.config_optimizer(scope)
            print('config optimizer in scope {}'.format(scope))
            self.train[scope] = self.optimizer[scope][0].apply_gradients(grads_and_vars[scope],\
                                                                        global_step=self.optimizer[scope][1])


        self.t = tf.Variable(0, trainable=False, name="iteration")
        self.increment_t = self.t.assign_add(1)
    

    
    def build_scope_loss_and_gradients(self,scope,vi_type,var_dict,losses,grads_and_vars):

        if vi_type in ['KLqp','KLqp_analytic']:
            losses[scope], grads_and_vars[scope] = self.build_reparam_ELBO_and_grads(scope,var_dict[scope],vi_type)

        elif vi_type in ['MAP','MLE']:
            losses[scope], grads_and_vars[scope] = self.build_MAP_MLE_and_grads(scope,var_dict[scope],vi_type)

        else:
            raise TypeError('Not supported vi type: '+vi_type)


    def build_loss_and_gradients(self,var_dict):
        
        losses = {}
        grads_and_vars = {}
        
        for scope in self.latent_vars.keys():
            vi_type = self.vi_types.get(scope,'KLqp')
            print(scope,vi_type)
            self.build_scope_loss_and_gradients(scope,vi_type,var_dict,losses,grads_and_vars)
        return losses, grads_and_vars


    def build_reparam_ELBO_and_grads(self,scope,var_list,vi_type):
        ll = 0.
        kl = 0.
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]      
        avg = 1./self.train_size

        # likelihood
        for x, qx in six.iteritems(data):
            ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))


        # KL-TERM for different inference methods       
        for z,qz in six.iteritems(latent_vars):
            if vi_type == 'KLqp':
                if isinstance(qz,RandomVariable):
                    kl += tf.reduce_sum(qz.log_prob(qz))-tf.reduce_sum(z.log_prob(qz))
                else:
                    raise TypeError('{} is not RandomVariable.')
                        
            
            elif vi_type == 'KLqp_analytic':                       
                kl += tf.reduce_sum(qz.kl_divergence(z))                
                                              
        kl *= avg
        loss = kl - ll + self.extra_loss.get(scope,0.)

        grads = tf.gradients(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars
    
    
    def build_MAP_MLE_and_grads(self,scope,var_list,vi_type):
        
        ll = 0.
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]

        for x, qx in six.iteritems(data):            
            ll += tf.reduce_sum(x.log_prob(qx))

        if 'vi_type' == 'MAP':
            for z,qz in six.iteritems(latent_vars):
                ll += tf.reduce_sum(z.log_prob(qz))
                
        loss = -ll + self.extra_loss.get(scope,0.)
        grads = tf.gradients(loss, var_list)
         

        grads_and_vars = list(zip(grads, var_list))

        self.grads = grads_and_vars # for debug

        return loss, grads_and_vars


    def config_optimizer(self,scope):

        # if not specified, config default optimizer #
        if not scope in self.optimizer:
            decay = (1000,0.9)
            with tf.variable_scope(scope):
                global_step = tf.Variable(0, trainable=False, name=scope+"_step")

            starter_learning_rate = 0.01
            learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                    global_step,
                                                    decay[0], decay[1], staircase=True)
            self.optimizer[scope] = (tf.train.AdamOptimizer(learning_rate),global_step)

        # if sepcified without step variable, generate one #
        elif len(self.optimizer[scope])==1:
            with tf.variable_scope(scope):
                global_step = tf.Variable(0, trainable=False, name=scope+"_step")

            self.optimizer[scope].append(global_step)

        return



    def update(self,scope,feed_dict=None,sess=None):
        if feed_dict is None:
            feed_dict = {}

        for key, value in six.iteritems(self.data[scope]):
            if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
                feed_dict[key] = value
        
        if not sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        _,t, loss = sess.run([self.train[scope], self.increment_t, self.losses[scope]], feed_dict)
        return {'t':t,'loss':loss}



    

    