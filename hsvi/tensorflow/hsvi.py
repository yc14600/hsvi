from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import six


from .distributions import RandomVariable


class Hierarchy_SVI:
    def __init__(self,latent_vars={},data={},scale={},optimizer={},vi_types={},\
                train_size=1000,extra_loss={},*args,**kwargs):
        """ Create an instance of Hierarchy_SVI. All the arguments are optional because they may need to be specified in different context according the model.

        Args:
            latent_vars (dict, optional): 
                latent random variables in each scope, including the prior (pz) and posterior (qz), e.g. {'global':{pz:qz,pw:qw}}. 
                Defaults to {}.
            data (dict, optional): 
                observations in each scope, including the random variable that represents the data (px) and observered samples (qx), e.g. {'global':{px:qx}}. 
                Defaults to {}.
            scale (dict, optional): 
                scale of observations in each scope, if not specified then it is 1 in all scopes. 
                Defaults to {}.
            optimizer (dict, optional): 
                optimizer in each scope, if not specified, an Adam optimizer will be configured for each scope.
                Defaults to {}.
            vi_types (dict, optional): 
                the type of inference methods in each scope, the supported types are [KLqp, KLqp_analytic, MAP, MLE], e.g. {'global':'KLqp'}. If not specified, it is set to KLqp in a scope. 
                Defaults to {}.
            train_size (int, optional): 
                training size of observations, it's used to reweight the KL term in VI when scale of observations is 1. 
                Defaults to 1000.
            extra_loss (dict, optional): 
                other losses in each scope, it's used to add custmoized losses other than the usual loss of VI, MAP, or MLE. 
                Defaults to {}.
        """
        
        print('start init hsvi')
        
        self.latent_vars = latent_vars
        self.data = data
        self.scopes = list(latent_vars.keys())
        self.scale = scale
        self.optimizer = optimizer
        self.vi_types = vi_types
        self.train_size = train_size 
        
        # to add any other loss 
        self.extra_loss = extra_loss

        # obtain variable list
        self.var_dict = {}               
        for scope in self.scopes:
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
                tmp.add(v)
            self.var_dict[scope] = list(tmp)
        
        self.losses, grads_and_vars = self._build_loss_and_gradients()       
        self.grads = grads_and_vars # add for debug

        self.train = {}
        for scope in self.scopes:
            self._config_optimizer(scope)
            print('config optimizer in scope {}'.format(scope))
            self.train[scope] = self.optimizer[scope][0].apply_gradients(grads_and_vars[scope],\
                                                                        global_step=self.optimizer[scope][1])


        self.t = tf.Variable(0, trainable=False, name="iteration")
        self.increment_t = self.t.assign_add(1)
    

    
    def _build_scope_loss_and_gradients(self,scope,vi_type,var_dict,losses,grads_and_vars):
        """build loss and gradients for a specific scope

        Args:
            scope (str): the scope name.
            vi_type (str): the typr of inference methods, the supported types are [KLqp, KLqp_analytic, MAP, MLE].
            var_dict (dict): the trainable variables of each scope.
            losses (dict): the built loss of each scope.
            grads_and_vars (dict): the variable gradients of each scope.

        Raises:
            TypeError: raise error when get not-supported vi_type
        """

        if vi_type in ['KLqp','KLqp_analytic']:
            losses[scope], grads_and_vars[scope] = self._build_reparam_ELBO_and_grads(scope,var_dict[scope],vi_type)

        elif vi_type in ['MAP','MLE']:
            losses[scope], grads_and_vars[scope] = self._build_MAP_MLE_and_grads(scope,var_dict[scope],vi_type)

        else:
            raise TypeError('Not supported vi type: '+vi_type)


    def _build_loss_and_gradients(self):
        """Build loss and gradients for all scopes

        Returns:
            tuple: the tuple of losses and gradients
        """
        
        losses = {}
        grads_and_vars = {}
        var_dict = self.var_dict
        
        for scope in self.latent_vars.keys():
            vi_type = self.vi_types.get(scope,'KLqp')
            print(scope,vi_type)
            self._build_scope_loss_and_gradients(scope,vi_type,var_dict,losses,grads_and_vars)
        return losses, grads_and_vars


    def _build_reparam_ELBO_and_grads(self,scope,var_list,vi_type):
        """Build ELBO loss of VI, which relies on reparameterization trick if the KL term is not analytic.


        Args:
            scope (str): the scope name
            var_list (list): the trainable variables in a specific scope
            vi_type (str): the type of VI, can be [KLqp, KLqp_analytic]. KLqp uses reparameterization trick for computing the KL term, KLqp_analytic uses the analytic form for computing the KL term.

        Raises:
            TypeError: raise error for qz that is not a random variable.

        Returns:
            tuple: a tuple of loss and gradients for a specific scope
        """
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
                    raise TypeError('{} is not RandomVariable.'.format(qz))
                        
            
            elif vi_type == 'KLqp_analytic':                       
                kl += tf.reduce_sum(qz.kl_divergence(z))                
                                              
        kl *= avg
        loss = kl - ll + self.extra_loss.get(scope,0.)

        grads = tf.gradients(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars
    
    
    def _build_MAP_MLE_and_grads(self,scope,var_list,vi_type):
        """Build MAP or MLE loss

        Args:
            scope (str): the scope name.
            var_list (list): the list of trainable variables in a specific scope.
            vi_type (str): the type of inference methods, can be ['MAP','MLE'].

        Returns:
            tuple: a tuple of loss and gradients for a specific scope
        """
        ll = 0.
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]

        for x, qx in six.iteritems(data):            
            ll += tf.reduce_sum(x.log_prob(qx))

        if vi_type == 'MAP':
            for z,qz in six.iteritems(latent_vars):
                ll += tf.reduce_sum(z.log_prob(qz))
                
        loss = -ll + self.extra_loss.get(scope,0.)
        grads = tf.gradients(loss, var_list)
         

        grads_and_vars = list(zip(grads, var_list))

        self.grads = grads_and_vars # for debug

        return loss, grads_and_vars


    def _config_optimizer(self,scope):
        """ Config default optimizer in a specified scope, which is Adam with learning_rate=0.01 and decay is stair case (1000,0.9).

        Args:
            scope (str): the scope name
        """

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
        """ Update step of parameters in one scope.

        Args:
            scope (str): 
                the scope name.
            feed_dict (dict, optional): 
                feed_dict for place holders in the specified scope.
            sess (tf.Session, optional):
                the session that contains the computation graph.

        Returns:
            dict: the update step and loss
        """
        if feed_dict is None:
            feed_dict = {}

        for key, value in six.iteritems(self.data[scope]):
            if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
                feed_dict[key] = value
        
        if not sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        _,t, loss = sess.run([self.train[scope], self.increment_t, self.losses[scope]], feed_dict)
        return {'t':t,'loss':loss}



    

    