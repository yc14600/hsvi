from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import six
import os

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence



class Hierarchy_SVI:

    def __init__(self,latent_vars={},data={},var_dict={},scale={},optimizer={},vi_types={},\
                    train_size=1000, extra_loss={},learning_rate={}, *args, **kwargs):
        """ Create an instance of Hierarchy_SVI. All the arguments are optional because they may need to be specified in different context according the model.

        Args:
            latent_vars (dict, optional): 
                latent random variables in each scope, including the prior (pz) and posterior (qz), e.g. {'global':{pz:qz,pw:qw}}. 
                Defaults to {}.
            data (dict, optional): 
                observations in each scope, including the random variable that represents the data (px) and observered samples (qx), e.g. {'global':{px:qx}}. 
                Defaults to {}.
            var_dict (dict, optional): 
                trainable variables in each scope, e.g. {'global':[var1, var2]}. 
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
        

        ### todo: check types of latent vars and data
        self.data = data
        self.latent_vars = latent_vars

        # need to manually input variables in different scopes,
        # because pytorch does not support variable scopes
        self.var_dict = var_dict 
        self.scopes = list(var_dict.keys())
        
        if scale is None:
            self.scale = {}
            for s in self.scopes:
                self.scale.update({s:{}})
        else:
            self.scale = scale

        self.optimizer = optimizer
        self.vi_types = vi_types
        self.train_size = train_size
        self.learning_rate = learning_rate
        

        # to add any other loss 
        self.extra_loss = extra_loss

        # obtain loss functions in each scope        
        self.losses = self._build_loss()  

        ### config optimizer for each scope ###
        for scope in self.scopes:
            self._config_optimizer(scope)        


    ### build loss for a specified scope ###
    def _build_scope_loss(self,scope,losses):
        """ Build loss for a specified scope.

        Args:
            scope (str): a specified scope
            losses (dict): the dict of loss functions

        Raises:
            TypeError: Not supported VI type.
        """
        vi_type = self.vi_types[scope]
        if vi_type in ['KLqp','KLqp_analytic']:
            losses[scope] = self._build_reparam_ELBO 

        elif vi_type in ['MAP','MLE']:
            losses[scope] = self._build_MAP_MLE
 
        else:
            raise TypeError('Not supported vi type: '+vi_type) 
        


    ### build loss for all scopes ###
    def _build_loss(self):
        """ Build loss for all scopes

        Returns:
            dict: the key is scope name,  the value is the loss built for the scope
        """
        
        losses = {}
        
        for scope in self.scopes:
            vi_type = self.vi_types.get(scope,'KLqp')
            print(scope,vi_type)
            self.vi_types[scope] = vi_type
            self._build_scope_loss(scope,losses)
        return losses

    
    @staticmethod
    def _build_reparam_ELBO(data,latent_vars,vi_type,train_size=1000,extra_loss=0.,scale={},*args,**kargs):
        """ Loss function of ELBO of VI, which relies on reparameterization trick if the KL term is not analytic.

        Args:
            data (dict): 
                observations in this scope, including the random variable that represents the data (px) and observered samples (qx), e.g. {px:qx}.
            latent_vars (dict): 
                latent random variables in this scope, including the prior (pz) and posterior (qz), e.g. {pz:qz,pw:qw}.
            vi_type (type):
                vi_type applied in this scope.
            train_size (int, optional): 
                training size of observations, it's used to reweight the KL term in VI when scale of observations is 1. 
                Defaults to 1000.
            extra_loss (scalar, optional): 
                other losses in this scope, it's used to add custmoized losses other than the usual loss of VI, MAP, or MLE. 
                Defaults to 0.
            scale (dict, optional): 
                scale of observations in each scope, if not specified then it is 1, e.g. {qx:5.}  
                Defaults to {}.

        Returns:
            scalar: final loss
        """
        ll = 0.
        kl = 0.       
        avg = 1./train_size

        # likelihood
        for x,qx in six.iteritems(data):
            ll += torch.mean(scale.get(x,1.)*x.log_prob(qx))
  

        # KL-TERM for different inference methods       
        for z,qz in six.iteritems(latent_vars):
            if vi_type == 'KLqp':                             
                kl += torch.sum(qz.log_prob(qz.rsample()))-torch.sum(z.log_prob(qz.rsample()))
                        
            elif vi_type == 'KLqp_analytic':  
                ## the function kl_divergence would require customized implementation  
                ## according to the specific distribution family of qz and z                   
                kl += torch.sum(kl_divergence(qz,z))                
                                              
        kl *= avg
        loss = kl - ll + extra_loss

        return loss
    

    @staticmethod
    def _build_MAP_MLE(data,latent_vars,vi_type,extra_loss=0.,*args,**kargs):
        """ Build loss function of MAP or MLE loss.

        Args:
            data (dict): 
                observations in this scope, including the random variable that represents the data (px) and observered samples (qx), e.g. {px:qx}.
            latent_vars (dict): 
                latent random variables in this scope, including the prior (pz) and posterior (qz), e.g. {pz:qz,pw:qw}.
            vi_type (type):
                vi_type applied in this scope.
            extra_loss (scalar, optional): 
                other losses in this scope, it's used to add customized losses other than the usual loss of VI, MAP, or MLE. 
                Defaults to 0.
        Returns:
            scalar: final loss
        """
        
        ll = 0.

        for x, qx in six.iteritems(data):
            ll += torch.mean(x.log_prob(qx))
        
        if vi_type == 'MAP':
            for z,qz in six.iteritems(latent_vars):
                ll += torch.mean(z.log_prob(qz.rsample()))
                
        loss = -ll + extra_loss
         
        return loss



    def _config_optimizer(self,scope):
        """ Config default optimizer for a specified scope.

        Args:
            scope (str): the scope name
        """

        # if not specified, config default optimizer #
        if not scope in self.optimizer:

            opt = torch.optim.Adam(self.var_dict[scope],lr=self.learning_rate.get(scope,0.001))

            scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=1000,gamma=0.9)
            self.optimizer[scope] = [opt,scheduler]
            
        return



    def update(self,scope,retain_graph=False):
        """ Update step of parameters in one scope.

        Args:
            scope (str): 
                the scope name.
            retain_graph (bool, optional): 
                whether or not to retain the graph after computing gradients each time. 
                Defaults to False.

        Returns:
            scalar: computed loss at the step
        """

        self.optimizer[scope][0].zero_grad()

        loss = self.losses[scope](self.data[scope],self.latent_vars[scope],self.vi_types[scope],\
                                    extra_loss=self.extra_loss.get(scope,0.),\
                                    scale=self.scale.get(scope,{}),train_size=self.train_size)
        
        loss.backward(retain_graph=retain_graph) 

        self.optimizer[scope][0].step()

        if len(self.optimizer[scope]) > 1:
            self.optimizer[scope][1].step()

        return loss




    

    