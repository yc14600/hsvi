from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import six
import os

from edward.inferences import Inference
from edward.models import RandomVariable,Normal
from .methods.svgd import SVGD


class Hierarchy_SVI(Inference):
    def __init__(self,latent_vars={},data={},*args,**kwargs):
        print('start init hsvi')
        super(Hierarchy_SVI,self).__init__(*args,**kwargs)
        self.latent_vars = latent_vars
        self.data = data
        self.scopes = list(latent_vars.keys())
        print('complete init hsvi')

    def initialize(self,scale={},optimizer={}, clipping={}, vi_types={}, constrain={},\
                    discriminator=None,loss_func={},samples={}, train_size=1000,\
                    task_id=0, coresets={},task_coef=[],renyi_alpha=0.,lamb=1.,trans_parm={}, \
                    extra_loss={}, *args, **kwargs):
        self.scale = scale
        self.optimizer = optimizer
        self.clipping = clipping
        self.vi_types = vi_types
        self.constrain = constrain
        self.discriminator = discriminator
        self.loss_func = loss_func
        self.coresets = coresets
        self.train_size = train_size   
        # lagrange multiplier for adjusting loss function
        self.lamb = lamb 
        # for cumulative kLqp
        self.task_id = task_id
        self.past_parms = None
        # for adaptive KLqp
        self.task_coef = task_coef       
        # for IWAE and Renyi
        self.samples = samples
        self.renyi_alpha = renyi_alpha
        # for gaussian transition natural gradient
        self.trans_parm = trans_parm
        # to add any other loss 
        self.extra_loss = extra_loss
        self.var_dict = {}               
        for scope in self.scopes:
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
                tmp.add(v)
            self.var_dict[scope] = list(tmp)
        
        self.losses, grads_and_vars = self.build_loss_and_gradients(self.var_dict)       
        self.grads = grads_and_vars #only for debug
        #self.config_trains(self.losses,grads_vars)
        self.train = {}
        for scope in self.scopes:
            self.config_optimizer(scope)
            #print('scope',scope)
            self.train[scope] = self.optimizer[scope][0].apply_gradients(grads_and_vars[scope],\
                                                                        global_step=self.optimizer[scope][1])

        super(Hierarchy_SVI,self).initialize(*args,**kwargs)
    
    def reinitialize(self,task_id=0,coresets={}):
        self.task_id = task_id
        self.coresets = coresets
        self.scopes = list(self.latent_vars.keys())
        for scope in self.scopes:
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
                tmp.add(v)
            self.var_dict[scope] = list(tmp)
        self.losses, grads_and_vars = self.build_loss_and_gradients(self.var_dict)       
        self.grads = grads_and_vars
        for scope in self.scopes:
            self.train[scope] = self.optimizer[scope][0].apply_gradients(grads_and_vars[scope],\
                                                                        global_step=self.optimizer[scope][1])

    def reinit_scope_train(self,scope,vi_type):
        self.build_scope_loss_and_gradients(scope,vi_type,self.var_dict,self.losses,self.grads)
        self.train[scope] = self.optimizer[scope][0].apply_gradients(self.grads[scope],\
                                                    global_step=self.optimizer[scope][1])
    
    def build_scope_loss_and_gradients(self,scope,vi_type,var_dict,losses,grads_and_vars):

        if vi_type in ['KLqp','Norm_flows','VAE','KLqp_analytic','KLqp_JS','KLqp_mutual','KLqp_GNG','KLqp_trans_GNG',\
                        'KLqp_analytic_GNG','KLqp_analytic_trans_GNG']:
            losses[scope], grads_and_vars[scope] = self.build_reparam_ELBO_and_grads(scope,var_dict[scope],vi_type)
            
        elif vi_type == 'cumulative_KLqp':
            losses[scope], grads_and_vars[scope] = self.build_cumulative_ELBO_and_grads(scope,var_dict[scope],vi_type)

        elif vi_type == 'KLqp_adaptive_prior':
            losses[scope], grads_and_vars[scope] = self.build_adaptive_ELBO_and_grads(scope,var_dict[scope])
        
        elif vi_type in ['IWAE','Renyi','IWAE_ll']:
            losses[scope], grads_and_vars[scope] = self.build_IWAE_and_grads(scope,var_dict[scope],vi_type)

        elif vi_type in ['MAP','MLE','MLE_GNG']:
            losses[scope], grads_and_vars[scope] = self.build_MAP_and_grads(scope,var_dict[scope],vi_type)

        # Stein_grads must be updated before Stein_VI, they are two steps of Stein-VI #
        elif vi_type == 'Stein_grads':
            losses[scope], grads_and_vars[scope] = self.build_ASVGD_and_grads(scope,var_dict[scope])

        elif vi_type == 'Inference_net':
            losses[scope], grads_and_vars[scope] = self.build_InfNet_and_grads(scope,var_dict[scope])

        elif vi_type in ['Implicit','Implicit_joint']:
            losses[scope+'/local'],losses[scope+'/global'],losses[scope+'/disc'], grads_and_vars[scope+'/local'], \
            grads_and_vars[scope+'/global'], grads_and_vars[scope+'/disc'] = self.build_Implicit_and_grads(scope,var_dict[scope],vi_type)

        elif vi_type in ['BCL_simple']:
            #losses[scope+'/task'],losses[scope+'/transition'],grads_and_vars[scope+'/task'],grads_and_vars[scope+'/transition']\
                losses[scope+'/task'],grads_and_vars[scope+'/task']\
                = self.build_BCL_simple_and_grads(scope,var_dict[scope])
        elif vi_type in ['BCL_simple2']:
            losses[scope+'/task'],losses[scope+'/transition'],grads_and_vars[scope+'/task'],grads_and_vars[scope+'/transition']\
                = self.build_BCL_simple2_and_grads(scope,var_dict[scope])
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
        coresets = self.coresets.get(scope,{})
        rec_err = 0.       
        avg = 1./self.train_size

        # likelihood
        for x, qx in six.iteritems(data):
            if isinstance(qx,tf.Tensor):
                qx_constrain = self.constrain.get(qx,None)
            else:
                qx_constrain = None

            if qx_constrain is None:
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))
            else:
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(tf.clip_by_value(qx,qx_constrain[0],qx_constrain[1])))
            if vi_type == 'VAE':
                rec_err += tf.reduce_mean(tf.square(x-qx))

        for x, qx in six.iteritems(coresets):           
            ll += tf.reduce_mean(x.log_prob(qx))*self.scale.get('coreset',1.)

        # KL-TERM for different inference methods       
        for z,qz in six.iteritems(latent_vars):
            if vi_type in ['KLqp','VAE','KLqp_GNG','KLqp_trans_GNG']:
                qz_constrain = self.constrain.get(qz,None)
        
                if qz_constrain is None:
                    if isinstance(qz,RandomVariable):
                        kl += tf.reduce_sum(qz.log_prob(qz))-tf.reduce_sum(z.log_prob(qz))
                    else:
                        kl += tf.reduce_sum(qz.log_prob(qz.value()))-tf.reduce_sum(z.log_prob(qz.value()))
                    #kl += tf.reduce_sum(qz.kl_divergence(z))
                else:
                    qz_samples = tf.clip_by_value(qz,qz_constrain[0],qz_constrain[1]) 
                    kl += tf.reduce_sum(qz.log_prob(qz_samples))-tf.reduce_sum(z.log_prob(qz_samples))
            

            elif vi_type == 'Norm_flows':
                # qz should be in the form of (z,lnq)
                kl = qz[1] - z.log_prob(qz[0])
            
            elif 'KLqp_analytic' in vi_type:                       
                kl += tf.reduce_sum(qz.kl_divergence(z))                
            
            elif vi_type == 'KLqp_JS' :  
                if self.task_id>0:                                   
                    kl += tf.reduce_sum((0.01*z.cross_entropy(qz)+0.99*qz.cross_entropy(z))-qz.entropy())
                else:
                    kl += tf.reduce_sum(qz.kl_divergence(z)) 
            
            elif vi_type == 'KLqp_mutual':
                kl += - 0.1*tf.reduce_sum(qz.log_prob(qz)) - 0.9*tf.reduce_sum(z.prob(qz)/qz.prob(qz)*z.log_prob(qz))   #-tf.reduce_mean(q.entropy())#
                                    
        kl *= avg
        loss = kl - ll + rec_err + self.extra_loss.get(scope,0.)#+ tf.losses.get_regularization_losses(scope=scope)
        self.kl = kl
        self.ll = ll
        grads_and_vars = []
        # analytic natural gradient of Normal distribution
        if 'GNG' in vi_type:
            if 'trans_GNG' in vi_type:
                grads_and_vars = self.natural_gradients_gaussian_trans(loss,scope)
            else:
                grads_and_vars = self.natural_gradients_gaussian(loss,scope)
            
            for V in six.itervalues(self.trans_parm[scope]):
                for v in V:
                    var_list.remove(v)
        #else:
        grads = tf.gradients(loss, var_list)
        if scope in self.clipping:
            print('clip',scope,self.clipping[scope])
            grads = [grd if grd is None else tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]
        grads_and_vars += list(zip(grads, var_list))

        return loss, grads_and_vars
    
    def build_cumulative_ELBO_and_grads(self,scope,var_list,vi_type):
        ll = 0.
        kl = 0.
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]        
        T = self.task_id
        coresets = self.coresets.get(scope,{})
        
        decay = 0.
        # likelihood
        for x, qx in six.iteritems(data):
            qx_constrain = self.constrain.get(qx,None)
            if qx_constrain is None:
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))
            else:
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(tf.clip_by_value(qx,qx_constrain[0],qx_constrain[1]))) 

        for x, qx in six.iteritems(coresets):            
                ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))
            

        if T > 0:
            #norm = np.sum([decay**(t) for t in range(T)])
            #print('norm',norm)
            avg = 1./(self.train_size)
            # cumulative KL-term
            if self.past_parms is None:
                self.past_parms = {}
            for z,qz in six.iteritems(latent_vars):
                if isinstance(z,Normal) and isinstance(qz,Normal):
                    
                    cum_z_parms = self.past_parms.get(qz,[0.]*4)
                   
                    # save cumulative past params
                    z_precision = 1./tf.square(z.scale)
                    dc = tf.exp(-decay)
                    # \sum_t^T \tau_t
                    cum_z_parms[0] *= dc
                    cum_z_parms[0] += z_precision
                    # \sum_t^T \tau_t * \mu_t 
                    cum_z_parms[1] *= dc
                    cum_z_parms[1] += z_precision * z.loc
                    # \sum_t^T \tau_t * \mu_t^2
                    cum_z_parms[2] *= dc
                    cum_z_parms[2] += z_precision * tf.square(z.loc)
                    # \sum_t^T \log \tau_t
                    cum_z_parms[3] *= dc
                    cum_z_parms[3] += tf.log(z_precision)
                    # save cumulative parameters
                    self.past_parms[qz] = cum_z_parms
                    
                    # cumulative kl      
                    #norm = tf.reduce_sum([tf.exp(-decay*t) for t in range(T)])   
                    norm = T       
                    kl +=  1./norm * 0.5 * tf.reduce_sum(tf.square(qz.loc) * cum_z_parms[0] - 2.* qz.loc * cum_z_parms[1] \
                                + cum_z_parms[2] + tf.square(qz.scale) * cum_z_parms[0] - cum_z_parms[3] \
                                + norm * tf.log(1./tf.square(qz.scale)) - norm) 
                                #+ (T-1)*0.5*tf.reduce_sum(1.+tf.log(2.*np.pi)+2.*tf.log(qz.scale))
                    
                    
        # first task
        else:
            avg = 1./(self.train_size)
            for z,qz in six.iteritems(latent_vars):
                kl += tf.reduce_sum(qz.kl_divergence(z))  

        kl *= avg
        loss = kl - ll 
        self.kl = kl
        self.ll = ll
        grads = tf.gradients(loss, var_list)

        if scope in self.clipping:
            grads = [tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]

        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars

    def build_adaptive_ELBO_and_grads(self,scope,var_list):
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]
        norm = tf.reduce_sum(self.task_coef)
        #norm = 1.
        ll=0.
        kl=0.
        for x, qx in six.iteritems(data):        
            ll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))

        for qz,cum_z_parms in six.iteritems(latent_vars):
            kl += 1./norm * 0.5 * tf.reduce_sum(tf.square(qz.loc) * cum_z_parms[0] - 2.* qz.loc * cum_z_parms[1] \
                    + cum_z_parms[2] + tf.square(qz.scale) * cum_z_parms[0] - cum_z_parms[3] \
                    + tf.log(1./tf.square(qz.scale)) - 1.)
        
        kl /= self.train_size
        loss = kl - ll 
        self.kl = kl
        self.ll = ll
        grads = tf.gradients(loss, var_list)

        if scope in self.clipping:
            grads = [tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]

        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars

    
    def build_IWAE_and_grads(self,scope,var_list,vi_type):
        
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]
        log_p = 0.
        log_q = 0.
        samples = self.samples
        data.update(self.coresets.get(scope,{}))
        if vi_type == 'IWAE':
            self.renyi_alpha = 0

        avg = 1./self.train_size
        #for i in range(n_samples):
        for z,qz in six.iteritems(latent_vars):
            qz_samples = samples[qz]
            dims = list(range(1,len(qz_samples.shape)))
            log_p += tf.reduce_sum(z.log_prob(qz_samples),axis=dims)
            log_q += tf.reduce_sum(qz.log_prob(qz_samples),axis=dims)
        log_p *= avg
        log_q *= avg
        self.kl = tf.reduce_mean(log_q - log_p) # only for performance analysis
        # NOTE: x is generated by multiple samples which are the same as in samples dict   
        self.ll = 0.
        for x,qx in six.iteritems(data):
            ll = x.log_prob(qx)
            dims = list(range(1,len(ll.shape)))
            self.ll += tf.reduce_mean(self.scale.get(x,1.)*ll) # only for performance analysis
            log_p += tf.reduce_sum(self.scale.get(x,1.)*ll,axis=dims)
        
      
        log_w = log_p - log_q       
        w = tf.exp(log_w-tf.reduce_logsumexp(log_w)) 
        
        if vi_type == 'IWAE_ll':
            log_w = self.ll
            w = tf.exp(log_w-tf.reduce_logsumexp(log_w))
            loss = self.kl - tf.reduce_sum(log_w * tf.stop_gradient(w))

        elif self.renyi_alpha!=0 and self.renyi_alpha!=1.:
            # Renyi-alpha
            log_w_alpha = (1.-self.renyi_alpha)*log_w
            w_alpha = tf.exp(log_w_alpha-tf.reduce_logsumexp(log_w_alpha))  
            loss = -tf.reduce_sum(log_w * tf.stop_gradient(w_alpha))
        else:
            # IWAE
            loss = -tf.reduce_sum(log_w * tf.stop_gradient(w))

        grads = tf.gradients(loss,var_list)

        if scope in self.clipping:
            grads = [tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]

        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars


    
    def build_MAP_and_grads(self,scope,var_list,vi_type):
        
        ll = 0.
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]

        for x, qx in six.iteritems(data):
            qx_constrain = self.constrain.get(qx,None)
            if qx_constrain is None:
                ll += tf.reduce_sum(self.scale.get(x,1.)*x.log_prob(qx))
            else:
                ll += tf.reduce_sum(self.scale.get(x,1.)*x.log_prob(tf.clip_by_value(qx,qx_constrain[0],qx_constrain[1])))
        if 'MLE' not in vi_type:
            for z,qz in six.iteritems(latent_vars):
                qz_constrain = self.constrain.get(qz,None)
                if qz_constrain is None:
                    ll += tf.reduce_sum(z.log_prob(qz))
                else:
                    ll += tf.reduce_sum(z.log_prob(tf.clip_by_value(qz,qz_constrain[0],qz_constrain[1])))

        loss = -ll + self.extra_loss.get(scope,0.)
        grads = tf.gradients(loss, var_list)
         
        self.ll = loss
        self.kl = tf.zeros([])
        if scope in self.clipping:
            grads = [tf.clip_by_value(grd,self.clipping[scope][0],self.clipping[scope][1]) for grd in grads]
        if vi_type == 'MLE_GNG':
            grads_and_vars = self.natural_gradients_gaussian(loss,scope)
        else:
            grads_and_vars = list(zip(grads, var_list))
        self.grads = grads_and_vars
        print('what!!')
        return loss, grads_and_vars
        
    def build_ASVGD_and_grads(self,scope,var_list):
        
        latent_vars = self.latent_vars[scope]
        data = self.data[scope]

        ll = 0.        
        # according to stein VAE equation (5)
        for x, qx in six.iteritems(data):
            qx = tf.expand_dims(qx,1)
            ll += tf.reduce_sum(self.scale.get(x,1.)*x.log_prob(qx),axis=0)
        dll = tf.gradients(ll,var_list)
        
        dlnp = []
        asvgd = SVGD()
        for z, qz in six.iteritems(latent_vars):
            lnp = z.log_prob(qz) 
            #loss += lnp - ll  
            dlnp += tf.gradients(lnp,var_list) # in case of different dimensions of qzs
        dlnp = [dx for dx in dlnp if dx is not None]
        grads = []
        for var,dll_i,dlnp_i in zip(var_list,dll,dlnp):   
            grads.append(asvgd.gradients(var,dll_i+dlnp_i))
    
        grads_and_vars = list(zip(grads, var_list))
        
        return tf.reduce_mean(ll), grads_and_vars

    def build_InfNet_and_grads(self,scope,var_list):

        data = self.data[scope]
        loss = 0.
        # qx is generated by an inference network, x is stein particles
        lossf = self.loss_func.get(scope,tf.losses.log_loss)
        for x, qx in six.iteritems(data):
            loss += tf.reduce_sum(lossf(x,qx)) 

        grads = tf.gradients(loss,var_list)
        grads_and_vars = list(zip(grads, var_list))

        return loss, grads_and_vars
    
    def build_Implicit_and_grads(self,scope,var_list,vi_type):
        
        data = self.data[scope]
        p_sample = {}
        q_sample = {}

        scale = 1.
        for x,qx in six.iteritems(data):
            if isinstance(x,tf.Tensor):
                p_sample[x] = x 
            elif isinstance(x,RandomVariable):
                p_sample[x] = x.value()
            else:
                raise TypeError('Not supported type of x!')

            if isinstance(qx,(tf.Tensor,tf.placeholder)):
                q_sample[x] = qx
            else:
                raise TypeError('Not supported type of qx!')
            
            scale *= self.scale.get(x,1.)
            

        # define ratios from discriminator
        if vi_type=='Implicit':
            with tf.variable_scope(scope+'/disc'):
                r_psamples = self.discriminator(p_sample)
                    
            with tf.variable_scope(scope+'/disc',reuse=True):
                r_qsamples = self.discriminator(q_sample)
        elif vi_type=='Implicit_joint':
            local_v = self.latent_vars[scope].get('local',None)
            global_v = self.latent_vars[scope].get('global',None)
            p_local_v ={}
            q_local_v = {}
            if local_v is not None:
                for z,qz in six.iteritems(local_v):
                    if isinstance(z,RandomVariable):
                        p_local_v[z] = z.value()
                    elif isinstance(z,tf.Tensor):
                        p_local_v[z] = z
                    else:
                        raise TypeError('Not supported type of z!')

                    if isinstance(qz,RandomVariable):
                        q_local_v[z] = qz.value()
                    elif isinstance(qz,tf.Tensor):
                        q_local_v[z] = qz
                    else:
                        raise TypeError('Not supported type of qz!')

            with tf.variable_scope(scope+'/disc'):
                r_psamples = self.discriminator(p_sample,p_local_v,global_v)
                    
            with tf.variable_scope(scope+'/disc',reuse=True):
                r_qsamples = self.discriminator(q_sample,q_local_v,global_v)
        
        ratio_loss = self.loss_func.get(scope,tf.losses.log_loss)
        loss_d = tf.reduce_mean(ratio_loss(tf.zeros_like(r_psamples),r_psamples) + \
                                ratio_loss(tf.ones_like(r_qsamples),r_qsamples))
        ll = tf.reduce_sum(r_psamples)*scale
        
        # split sub-scopes
        loss={}
        grads={}
        grads_and_vars={}
        for subscope in ['local','global','disc']:
            kl = 0.               
            new_scope = '/'.join([scope,subscope])
            
            #print(new_scope)
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=new_scope):              
                tmp.add(v)
            svar_list = list(tmp)
            if len(svar_list)<1:
                continue
            

            if subscope == 'disc':
                grads[subscope] = tf.gradients(loss_d, svar_list)
                grads_and_vars[subscope] = list(zip(grads[subscope], svar_list))
                self.scopes.append(new_scope)
            else:
                # Implicit_joint doesn't include local variable KL-divergence
                if subscope == 'global' or vi_type == 'Implicit':
                    latent_vars = self.latent_vars[scope][subscope]
                    for z,qz in six.iteritems(latent_vars):
                        qz_constrain = self.constrain.get(qz,None)
                        if qz_constrain is None:
                            kl += tf.reduce_sum(qz.log_prob(qz))-tf.reduce_sum(z.log_prob(qz))
                        else:
                            qz_samples = tf.clip_by_value(qz,qz_constrain[0],qz_constrain[1]) 
                            kl += tf.reduce_sum(qz.log_prob(qz_samples))-tf.reduce_sum(z.log_prob(qz_samples))

                loss[subscope] = kl - ll
                grads[subscope] = tf.gradients(loss[subscope], svar_list)
                grads_and_vars[subscope] = list(zip(grads[subscope], svar_list))
                self.scopes.append(new_scope)
            
            #print(subscope,grads_and_vars[subscope],svar_list)

        self.scopes.remove(scope)
        return loss.get('local',None),loss.get('global',None),loss_d,grads_and_vars.get('local',None),grads_and_vars.get('global',None),grads_and_vars['disc']
    
    # build loss function for Bayesian Continual Learning simple version
    def build_BCL_simple_and_grads(self,scope,var_list):
        data = self.data[scope]
        latent_vars = self.latent_vars[scope]
        #coresets = self.coresets.get(scope,{})
        # split sub-scopes
        loss={}
        grads={}
        grads_and_vars={}
        data_ll = 0.
        data_logll = 0.
        ll = 0.    
        tll = 0.
       
        for x,qx in six.iteritems(data['current_parm']):
            data_logll += tf.reduce_mean(self.scale.get(x,1.)*x.log_prob(qx))
        #for x,qx in six.iteritems(data['prev_parm']):
        #    data_ll += tf.reduce_mean(tf.exp(self.scale.get(x,1.))*x.prob(qx))

        for q,tq in six.iteritems(latent_vars):
            ll += tf.reduce_sum(q.prob(tq)/tq.prob(tq)*q.log_prob(tq))    #-tf.reduce_mean(q.entropy())#
            tll += tf.reduce_sum(tq.prob(q)/q.prob(q)*tq.log_prob(q)) #0.8*tf.reduce_mean(tq.prob(q)/q.prob(q)*tq.log_prob(q))+0.2*tf.reduce_mean(tq.entropy())
        #tll *= data_ll
        #self.data_ll = data_ll
        self.data_logll = data_logll
        self.ll = ll
        self.tll = -self.lamb*tll

        for subscope in ['task']:
                       
            new_scope = '/'.join([scope,subscope])
            
            #print(new_scope)
            # get variables of subscope
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=new_scope):              
                tmp.add(v)
            svar_list = list(tmp)
            if len(svar_list)<1:
                continue

            if subscope == 'task':                              
                loss[subscope] = self.lamb *(ll  -  tll )/self.train_size - data_logll

            elif subscope == 'transition':
                loss[subscope] = - self.lamb * tll

            grads[subscope] = tf.gradients(loss[subscope], svar_list)
            grads_and_vars[subscope] = list(zip(grads[subscope], svar_list))
            self.scopes.append(new_scope)
        self.scopes.remove(scope)
        #return loss['task'],loss['transition'],grads_and_vars['task'],grads_and_vars['transition']
        return loss['task'],grads_and_vars['task']

    def build_BCL_simple2_and_grads(self,scope,var_list):
        data = self.data[scope]
        latent_vars = self.latent_vars[scope]
        #coresets = self.coresets.get(scope,{})
        # split sub-scopes
        loss={}
        grads={}
        grads_and_vars={}
        #data_ll = 0.
        data_logll = 0.
        data_clogll = 0.
        #ll = 0.    
        kl = 0.
       
        for x,qx in six.iteritems(data['current_parm']):
            data_logll += tf.reduce_mean(x.log_prob(qx))
        #for x,qx in six.iteritems(data['prev']):
        #    data_logll += tf.reduce_mean(x.log_prob(qx))
        for x,qx in six.iteritems(data['prev_parm']):
            data_clogll += tf.reduce_mean(x.log_prob(qx))

        for q,tq in six.iteritems(latent_vars):
            #ll += -tf.reduce_mean(q.entropy())    
            #tll += tf.reduce_mean(tq.prob(q)/q.prob(q)*tq.log_prob(q)) #0.8*tf.reduce_mean(tq.prob(q)/q.prob(q)*tq.log_prob(q))+0.2*tf.reduce_mean(tq.entropy())
            kl += tf.reduce_sum(q.kl_divergence(tq))
        #tll *= data_ll
        #self.data_wlogll = data_wlogll
        kl /= self.train_size
        self.kl = kl
        self.ll = data_logll+data_logll

        for subscope in ['task','transition']:
                       
            new_scope = '/'.join([scope,subscope])
            
            #print(new_scope)
            # get variables of subscope
            tmp = set()
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=new_scope):              
                tmp.add(v)
            svar_list = list(tmp)
            if len(svar_list)<1:
                continue

            if subscope == 'task':                              
                loss[subscope] = kl  - data_logll 
                grads[subscope] = tf.gradients(loss[subscope], svar_list)
                grads_and_vars[subscope] = self.natural_gradients_gaussian(loss[subscope],scope)

            elif subscope == 'transition':
                loss[subscope] = kl - data_clogll
                grads[subscope] = tf.gradients(loss[subscope], svar_list)
               
                grads_and_vars[subscope] = list(zip(grads[subscope], svar_list))

            self.scopes.append(new_scope)
        self.scopes.remove(scope)
        return loss['task'],loss['transition'],grads_and_vars['task'],grads_and_vars['transition']       
    
    # configure default optimizer
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
        # get parent scope to fetch data for implicit vi
        if '/' in scope:
            pscope = scope.split('/')[0]
        else:
            pscope = scope

        for key, value in six.iteritems(self.data[pscope]):
          if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
            feed_dict[key] = value
        
        if not sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))   
        _,t, loss = sess.run([self.train[scope], self.increment_t, self.losses[scope]], feed_dict)
        return {'t':t,'loss':loss}


    def print_progress(self, info_dict):
        """Print progress to output."""
        if self.n_print != 0:
            t = info_dict['t']
            if t == 1 or t % self.n_print == 0:
                self.progbar.update(t, {'Loss': info_dict['loss']})

    # apply natural gradients to variables of a Gaussian distribution,
    # and variables of the Gaussian distribution must be defined as \mu,\log \sigma
    def natural_gradients_gaussian(self,loss,scope):
        print('generate NG')
        trans_parm = self.trans_parm[scope]

        grads_and_vars =[]
        for qz_vars in six.itervalues(trans_parm):
            g = tf.gradients(loss,qz_vars)  

            if g[0] is not None:        
                g[0] *= tf.exp(2.*qz_vars[1])
                grads_and_vars.append((g[0],qz_vars[0]))
            if g[1] is not None:
                g[1] *= 0.5            
                grads_and_vars.append((g[1],qz_vars[1]))        
        
        return grads_and_vars

    def natural_gradients_gaussian_trans(self,loss,scope):
        grads_and_vars =[]
        trans_parm = self.trans_parm[scope]
        print('generate NG')
        #print(trans_parm)
        latent_vars = self.latent_vars[scope]
        #print(latent_vars)
        for z,qz in six.iteritems(latent_vars):
            parms = trans_parm.get(qz,None)
            if parms is None:
                continue
            mu_0 = z.loc
            v_0 = tf.square(z.scale)
            #mu = qz.loc
            v = tf.square(qz.scale)
            g = tf.gradients(loss,parms)
            # Fisher of A
            fa = 3.*v_0*tf.square(parms[0])/tf.square(v) + tf.square(mu_0)/v
            ng_a = g[0]/fa
            grads_and_vars.append((ng_a,parms[0]))
            # Fisher of B
            ng_b = g[2]*v
            grads_and_vars.append((ng_b,parms[2]))
            # Fisher of O
            fo = 2.*tf.exp(2.*parms[1])/tf.square(v)
            ng_o = g[1]/fo
            grads_and_vars.append((ng_o,parms[1]))

        return grads_and_vars
