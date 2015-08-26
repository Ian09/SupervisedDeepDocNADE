'''

# Copyright 2015 Yin Zheng, Yu-Jin Zhang, Hugo Larochelle. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Yin Zheng, Yu-Jin Zhang, Hugo Larochelle ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Yin Zheng, Yu-Jin Zhang, Hugo Larochelle OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Yin Zheng, Yu-Jin Zhang, Hugo Larochelle.

Created on Aug 15, 2015

@author: zhengyin

@contact: yzheng3xg@gmail.com

@summary: The class for paper A Deep and Autoregressive Approach for Topic Modeling of Multimodal Data, TPAMI 2015

'''

import theano
import theano.tensor as T
import theano.sandbox.linalg as Tlin
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams as RS_FixationNADE
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sparse as S
import Image
import numpy as np
import copy as cp
import scipy.sparse as sp
import gc
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import sys 
sys.setrecursionlimit(50000) 
# import pydot
import warnings
try:
    from scipy.linalg import cholesky, det, solve
except ImportError:
    warnings.warn("Could not import some scipy.linalg functions")
# import theano.tensor as T
from theano import config

activation_functions = {"sigmoid": theano.tensor.nnet.sigmoid, "reclin": lambda x: theano.tensor.maximum(x, 0.0), "tanh": theano.tensor.tanh}
class DeepDocNADE(object):
    ''' Theano verson for deep DocNADE'''
    
    def __init__(self,
                 hidden_size = [100,100],
                 learning_rate = 0.001,
                 activation_function = 'sigmoid',
                 testing_ensemble_size = 1,
                 hidden_bias_scaled_by_document_size = False,
                 word_representation_size = 0,
                 seed_np = 1234,
                 seed_theano = 4321, 
                 use_dropout = False, 
                 dropout_rate = [0.5],
                 normalize_by_document_size = False,
                 anno_weight = 1.0,
                 global_feature_weight = 1.0,
                 batch_size = 1,
                 aver_words_count = 1,
                 preprocess_method = 'std',
                 decrease_constant = 0.999,
                 length_limit = 15.0,
                 polyakexp_weight = 0.99 
                 
                 ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.activation_function_name = activation_function
        self.aver_words_count = aver_words_count
        self.testing_ensemble_size = testing_ensemble_size
        self.hidden_bias_scaled_by_document_size = hidden_bias_scaled_by_document_size
        self.seed_np = seed_np
        self.seed_theano = seed_theano
#         self.seed_shuffle = seed_shuffle
        self.word_representation_size = word_representation_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.normalize_by_document_size = normalize_by_document_size
        self.n_layers = len(self.hidden_size)
        self.anno_weight = anno_weight
        self.global_feature_weight = global_feature_weight
        self.batch_size = batch_size
        self.preprocess_method = preprocess_method
        self.decrease_constant = decrease_constant
        dec_learning_rate_value = np.asarray(learning_rate, dtype=theano.config.floatX)
        self.dec_learning_rate = theano.shared(value=dec_learning_rate_value, name='dec_learning_rate')
        self.length_limit = length_limit
        self.polyakexp_weight = polyakexp_weight
        
          
    def initialize(self,voc_size, anno_voc_size, global_feature_size, region_split):
        
        self.activation = activation_functions[self.activation_function_name]
        self.rng_theano = RandomStreams(seed=self.seed_theano)
        self.rng = np.random.mtrand.RandomState(self.seed_np) 
        self.voc_size = voc_size
        self.anno_voc_size = anno_voc_size
        self.global_feat_size = global_feature_size
        self.region_split = region_split
        
        
        self.W = []
        self.c = []
        input_size = self.voc_size + self.anno_voc_size
        for hidden_size in self.hidden_size:
#             W_value = (2*self.rng.rand(input_size,hidden_size)-1)/(np.max([input_size, hidden_size]))
#             W_value = self.rng.uniform(-np.sqrt(6)/(input_size + hidden_size), np.sqrt(6)/(input_size + hidden_size), size=(input_size, hidden_size))
            W_value = self.rng.uniform(-np.sqrt(6)/np.sqrt(input_size + hidden_size), np.sqrt(6)/np.sqrt(input_size + hidden_size), size=(input_size, hidden_size))
            W_value = np.asarray(W_value, theano.config.floatX)
            c_value = np.zeros((hidden_size,),theano.config.floatX)
            W = theano.shared(value = W_value, name = 'W')
            c = theano.shared(value = c_value, name = 'c')
            self.W.append(W)
            self.c.append(c)
            input_size = hidden_size
#         G_value = (2*self.rng.rand(self.global_feat_size,self.hidden_size[0])-1)/(np.max([self.global_feat_size, self.hidden_size[0]]))
#         G_value = self.rng.uniform(-np.sqrt(6)/(self.global_feat_size + self.hidden_size[0]), np.sqrt(6)/(self.global_feat_size + self.hidden_size[0]), size=(self.global_feat_size, self.hidden_size[0]))
        G_value = self.rng.uniform(-np.sqrt(6)/np.sqrt(self.global_feat_size + self.hidden_size[0]), np.sqrt(6)/np.sqrt(self.global_feat_size + self.hidden_size[0]), size=(self.global_feat_size, self.hidden_size[0]))
        G_value = np.asarray(G_value, theano.config.floatX)
        self.G = theano.shared(value=G_value, name = 'G')
        
        anno_mask = np.ones((self.batch_size, self.voc_size+self.anno_voc_size), theano.config.floatX)
        anno_mask[:, -self.anno_voc_size:] = self.anno_weight
        self.anno_mask = theano.shared(value=anno_mask, name='anno_mask')
        
        self.W_polyak = cp.deepcopy(self.W) 
        self.c_polyak = cp.deepcopy(self.c)
        self.G_polyak = cp.deepcopy(self.G)   
    def __deepcopy__(self,memo): 
        print "Warning: the deepcopy only copies the parameters, you SHOULD call compile_function for the functions"
        newone = type(self)()
        memo[id(self)] = newone
        old_dict = dict(self.__dict__)
        for key,val in old_dict.items():
            if key in ['train','valid','test']:
                print 'escape %s'%(key)
                pass
            else:
                newone.__dict__[key] = cp.deepcopy(val, memo)
        return newone  
    
      
    def build_graph(self, debug, hist_visual, hist_anno, global_feature, n_layer_to_build, W, c, V, b, G, flag_train):
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
            
        
        hist_anno_dense = hist_anno.toarray()
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)
#         anno_mask = T.ones(hist.shape, theano.config.floatX)
#         tt = T.ones((hist.shape[0],2000), theano.config.floatX)*self.anno_weight
#         anno_weighted_mask = T.set_subtensor(anno_mask[:, -2000:], tt)
        if debug==True:
            mask_unif = 0.5*T.ones(shape=hist.shape, dtype=theano.config.floatX)
            
        else:
            mask_unif = 1.0 - self.rng_theano.uniform(size=hist.shape, low=0., high=1., dtype=theano.config.floatX)
        mask_counts = mask_unif*(hist+1)
        input = T.floor(mask_counts)*self.anno_mask
        hist = hist*self.anno_mask
        d = input.sum(axis = 1)
        D = hist.sum(axis = 1)
        predict = hist - input
        condition_bias = T.dot(global_feature, G)
        
        if self.preprocess_method == 'None':
            tmp_input = input
        elif self.preprocess_method == 'std':
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
        else:
            print 'Unknow preprocess method'
            exit(-1)
        
#         tmp_input = input
        for i in xrange(n_layer_to_build):
            if i==0:
                
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)
            else:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i])*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i])*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i])
            tmp_input = h
        log_prob_each_word = T.log(T.nnet.softmax_with_bias(T.dot(h, V),b)+1e-16)
        log_prob_each_bin = log_prob_each_word*predict
        nll = -log_prob_each_bin.sum(axis=1)
        #TODO:??divide D??
        if self.normalize_by_document_size:
            cost = T.mean(1.0/(D-d)*nll)
        else:
            cost = T.mean(D/(D-d)/self.aver_words_count*nll)
        
        return cost,h,mask_unif,mask_counts,input,predict,hist,log_prob_each_bin,nll,condition_bias
    
    def build_compute_representation_graph(self, hist_visual, hist_anno, global_feature, n_layer_to_build, W, c,G, flag_train):
        
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
             
         
        hist_anno_dense = hist_anno.toarray()
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)
#         anno_mask = T.ones(hist.shape, theano.config.floatX)
#         anno_weighted_mask = T.set_subtensor(anno_mask[:, -self.anno_voc_size:], self.anno_weight)
        if self.preprocess_method == 'None':
            input = hist*self.anno_mask
            tmp_input = input
        elif self.preprocess_method == 'std':
            input = hist*self.anno_mask
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            input = hist*self.anno_mask
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
#             squared_input = input**2
#             init_tmp_input = T.ones(shape=input.shape, dtype=theano.config.floatX)
#             last_rsp = 0
#             for r_sp in self.region_split:
#                 div_number = T.sqrt(squared_input[:,last_rsp:r_sp].sum(axis=1))
#                 tmp_input = T.set_subtensor(init_tmp_input[:,last_rsp:r_sp], input[:,last_rsp:r_sp]/(div_number[:, np.newaxis]+1e-16))
#                 init_tmp_input = tmp_input
#                 last_rsp = r_sp
#             anno_factor = tmp_input[:,:self.region_split[-2]].sum(axis=1)
#             tmp_input = T.set_subtensor(tmp_input[:,self.region_split[-2]:], tmp_input[:,self.region_split[-2]:]*anno_factor[:, np.newaxis]*2)
        else:
            print 'Unknow preprocess method'
            exit(-1)
#         input = hist




        condition_bias = T.dot(global_feature, G)
#           
#        tmp_input = input
        for i in xrange(n_layer_to_build):
            if i==0:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)
            else:
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i])
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i])*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i])*(1.-self.dropout_rate[i]))
            tmp_input = h
             
         
        return h
    
    def compile_function(self, n_layers, trainset, validset):
        
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.cast(T.scalar('index'), 'int32')
        flag_train = T.scalar(name='flag_train')
        cost,hidden_representation,mask_unif,mask_counts,input,predict,hist,log_prob_each_bin,nll,condition_bias = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G, flag_train)
        
        params = [self.V, self.b, self.G]
        params.extend(self.W[:n_layers])
        params.extend(self.c[:n_layers])
        
        polyaks = [self.V_polyak, self.b_polyak, self.G_polyak]
        polyaks.extend(self.W_polyak[:n_layers])
        polyaks.extend(self.c_polyak[:n_layers])
        
        params_gradient = [T.grad(cost, param) for param in params]
        
        
        updates = []

        for param, param_gradient, polyak in zip(params, params_gradient, polyaks):
            param_updated = param - self.dec_learning_rate*param_gradient
            if param.get_value(borrow=True).ndim==2:
                col_norms = T.sqrt(T.sum(T.sqr(param_updated), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.length_limit))
                col_scale = desired_norms / (1e-16 + col_norms)
                updates.append((param, param_updated*col_scale))
            else:
                updates.append((param, param_updated))
                
            polyak_updated = self.polyakexp_weight*polyak + (1-self.polyakexp_weight)* param_updated      
            updates.append((polyak, polyak_updated))
            
            
        updates.append((self.dec_learning_rate, self.dec_learning_rate*self.decrease_constant))
        
            
        self.train = theano.function(inputs = [index],
                                    updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:trainset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:trainset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:trainset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(1,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )   
#         theano.printing.pydotprint(self.train, outfile='/home/local/USHERBROOKE/zhey2402/DeepDocNADE/pic.png')
        self.valid = theano.function(inputs = [index],
#                                      updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:validset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:validset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:validset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(0,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )
           
        
    def compile_compute_representation_function(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
#         cost,hidden_representation,input, anno_weighted_mask = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G)
        hidden_representation = self.build_compute_representation_graph(hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.G, flag_train)
        self.compute_representation = theano.function(inputs = [index],
                                                      outputs = hidden_representation,  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
#                                                       mode='DebugMode'
                                                     ) 
    def compile_compute_representation_function_polyak(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
#         cost,hidden_representation,input, anno_weighted_mask = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G)
        hidden_representation = self.build_compute_representation_graph(hist_visual, hist_anno, global_feature, n_layers, self.W_polyak, self.c_polyak, self.G_polyak, flag_train)
        self.compute_representation = theano.function(inputs = [index],
                                                      outputs = hidden_representation,  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
#                                                       mode='DebugMode'
                                                     )
    def compile_LayerByLayer_function(self, n_layers, trainset, validset):
        
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.cast(T.scalar('index'), 'int32')
        flag_train = T.scalar(name='flag_train')
        cost,hidden_representation = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G, flag_train)
        
        params = [self.V, self.b, self.G, self.W[n_layers-1], self.c[n_layers-1]]
        polyaks = [self.V_polyak, self.b_polyak, self.G_polyak, self.W_polyak[n_layers-1], self.c_polyak[n_layers-1]]
        params_gradient = [T.grad(cost, param) for param in params]
        
        
        updates = []

        for param, param_gradient, polyak in zip(params, params_gradient, polyaks):
            param_updated = param - self.dec_learning_rate*param_gradient
            if param.get_value(borrow=True).ndim==2:
                col_norms = T.sqrt(T.sum(T.sqr(param_updated), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.length_limit))
                col_scale = desired_norms / (1e-16 + col_norms)
                updates.append((param, param_updated*col_scale))
            else:
                updates.append((param, param_updated))
                
            polyak_updated = self.polyakexp_weight*polyak + (1-self.polyakexp_weight)* param_updated      
            updates.append((polyak, polyak_updated))
        updates.append((self.dec_learning_rate, self.dec_learning_rate*self.decrease_constant))
            
        self.train = theano.function(inputs = [index],
                                    updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:trainset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:trainset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:trainset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(1,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )   
#         theano.printing.pydotprint(self.train, outfile='/home/local/USHERBROOKE/zhey2402/DeepDocNADE/pic.png')
        self.valid = theano.function(inputs = [index],
#                                      updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:validset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:validset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:validset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(0,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )  
    
              
    def verify_gradients(self):
        
        def fun(W0,W1, c0, c1, V, b,G):
            
            hist_visual = np.array([[1,2,3,4,5,6,7,8,9,0]], dtype = theano.config.floatX)
            hist_anno = sp.csr_matrix([[0,0,1,0,0,0,1,2,0,0]], dtype = theano.config.floatX)
            global_feature = np.array([[1,2,3,4,5,6,7,8,9,0]], dtype = theano.config.floatX)
            n_layers = 2
            cost, h=self.build_graph(True, hist_visual, hist_anno, global_feature, n_layers, 
                                  [W0,W1], [c0,c1], V, b, G)
                
            return cost
        print 'Warning: verify_gradient will reinitialize the model!!!'
        self.hidden_size = [100,100]
        self.n_classes = 7
        self.dropout_rate = 0.5
        self.activation = activation_functions['reclin']
        self.n_layers = len(self.hidden_size)
        self.initialize(10,10,10)
        rng = np.random.RandomState(42)
        
        
#         rng = np.random.RandomState(42)
        theano.tests.unittest_tools.verify_grad(fun, [self.W[0].get_value(), self.W[1].get_value(),self.c[0].get_value(), self.c[1].get_value(),
                                                      self.V.get_value(), self.b.get_value(), self.G.get_value()], rng = rng)
        
        
        
        
    def remove_activation(self):
        
        del self.activation
        
    def add_activation(self):
        
        self.activation = activation_functions[self.activation_function_name]
        
    def remove_top_layer(self):
        if hasattr(self, 'V'):
            del self.V
        if hasattr(self, 'b'):
            del self.b
    
    def add_top_layer(self, layer_id):
        '''
        layer_id is the id of the hidden layer (starting from 0) on which we build the top layer to compute the conditionals
        '''
        if layer_id <0:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if layer_id > self.n_layers-1:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
#         V_value = (2*self.rng.rand(self.hidden_size[layer_id],self.voc_size+self.anno_voc_size)-1)/(np.max([self.voc_size+self.anno_voc_size, self.hidden_size[layer_id]])) 
#         V_value = self.rng.uniform(-np.sqrt(6)/(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), np.sqrt(6)/(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), size=( self.hidden_size[layer_id],self.voc_size+self.anno_voc_size))
        V_value = self.rng.uniform(-np.sqrt(6)/np.sqrt(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), np.sqrt(6)/np.sqrt(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), size=( self.hidden_size[layer_id],self.voc_size+self.anno_voc_size))
        V_value = np.asarray(V_value, theano.config.floatX)
        self.V = theano.shared(value = V_value, name = 'V')
        b_value = np.zeros((self.voc_size+self.anno_voc_size), theano.config.floatX)
        self.b = theano.shared(value = b_value, name = 'b')
        self.V_polyak = cp.deepcopy(self.V)
        self.b_polyak = cp.deepcopy(self.b)
        
    def copy_parameters(self, source):
        
        self.V.set_value(source.V.get_value())
        self.b.set_value(source.b.get_value())
        self.V_polyak.set_value(source.V_polyak.get_value())
        self.b_polyak.set_value(source.b_polyak.get_value())
        for i in xrange(self.n_layers):
            self.W[i].set_value(source.W[i].get_value())
            self.c[i].set_value(source.c[i].get_value())
            self.W_polyak[i].set_value(source.W_polyak[i].get_value())
            self.c_polyak[i].set_value(source.c_polyak[i].get_value())
        self.G.set_value(source.G.get_value())
        self.G_polyak.set_value(source.G_polyak.get_value())
        self.dec_learning_rate.set_value(source.dec_learning_rate.get_value())
        
        
        
class SupDeepDocNADE(object):
    ''' Theano verson for Supervised deep DocNADE'''
    
    def __init__(self,
                 hidden_size = [100,100],
                 learning_rate = 0.001,
                 learning_rate_unsup = 0.001,
                 activation_function = 'sigmoid',
                 testing_ensemble_size = 1,
                 hidden_bias_scaled_by_document_size = False,
                 word_representation_size = 0,
                 seed_np = 1234,
                 seed_theano = 4321, 
                 use_dropout = False, 
                 dropout_rate = [0.5],
                 normalize_by_document_size = False,
                 anno_weight = 1.0,
                 global_feature_weight = 1.0,
                 batch_size = 1,
                 unsup_weight = 0.001,
                 sup_option = 'full',
                 aver_words_count = 1,
                 n_connection = 15,
                 bias = 0.0,
                 rescale = 0.01,
                 preprocess_method = 'SPM',
                 decrease_constant = 0.999,
                 length_limit = 15.0,
                 polyakexp_weight = 0.99
                 
                 ):
        self.n_epoches_trained = 0
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.learning_rate_unsup = learning_rate_unsup
        self.activation_function_name = activation_function
        self.aver_words_count = aver_words_count
        self.testing_ensemble_size = testing_ensemble_size
        self.hidden_bias_scaled_by_document_size = hidden_bias_scaled_by_document_size
        self.seed_np = seed_np
        self.seed_theano = seed_theano
#         self.seed_shuffle = seed_shuffle
        self.word_representation_size = word_representation_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.normalize_by_document_size = normalize_by_document_size
        self.n_layers = len(self.hidden_size)
        self.anno_weight = anno_weight
        self.global_feature_weight = global_feature_weight
        self.batch_size = batch_size
        self.unsup_weight = unsup_weight
#         self.unsup_weight = theano.shared(value=unsup_weight,name='unsup')
        self.sup_option = sup_option
        self.n_connection = n_connection
        self.bias = bias
        self.rescale = rescale
        self.preprocess_method = preprocess_method
        self.decrease_constant = decrease_constant
        dec_learning_rate_value = np.asarray(learning_rate, dtype=theano.config.floatX)
        self.dec_learning_rate = theano.shared(value=dec_learning_rate_value, name='dec_learning_rate')
        self.length_limit = length_limit
        self.polyakexp_weight = polyakexp_weight
        
          
    def initialize(self,voc_size, anno_voc_size, global_feature_size, n_classes, region_split):
        
        self.activation = activation_functions[self.activation_function_name]
        self.rng_theano = RandomStreams(seed=self.seed_theano)
        self.rng = np.random.mtrand.RandomState(self.seed_np) 
#         self.rng = np.random.mtrand.RandomState(self.seed)
#         self.rng_shuffle = np.random.mtrand.RandomState(self.seed_shuffle)
        self.voc_size = voc_size
        self.anno_voc_size = anno_voc_size
        self.global_feat_size = global_feature_size
        self.n_classes = n_classes
        self.region_split = region_split
        
        
        
        
        self.W = []
        self.c = []
        input_size = self.voc_size + self.anno_voc_size
        cnt = 1
        for hidden_size in self.hidden_size:
            W_value = 1*self.rng.uniform(-np.sqrt(6)/np.sqrt(input_size + hidden_size), np.sqrt(6)/np.sqrt(input_size + hidden_size), size=(input_size, hidden_size))
#             W_value = 10*generate_SparseConnectionMat(self.rng, input_size, hidden_size, self.n_connection, self.rescale, self.bias)*cnt
            W_value = np.asarray(W_value, theano.config.floatX)
            c_value = np.zeros((hidden_size,),theano.config.floatX)
            W = theano.shared(value = W_value, name = 'W')
            c = theano.shared(value = c_value, name = 'c')
            self.W.append(W)
            self.c.append(c)
            input_size = hidden_size
            cnt *= 3
            
        G_value = self.rng.uniform(-np.sqrt(6)/np.sqrt(self.global_feat_size + self.hidden_size[0]), np.sqrt(6)/np.sqrt(self.global_feat_size + self.hidden_size[0]), size=(self.global_feat_size, self.hidden_size[0]))
        G_value = np.asarray(G_value, theano.config.floatX)
        self.G = theano.shared(value=G_value, name = 'G')
        
        anno_mask = np.ones((self.batch_size, self.voc_size+self.anno_voc_size), theano.config.floatX)
        anno_mask[:, -self.anno_voc_size:] = self.anno_weight
        self.anno_mask = theano.shared(value=anno_mask, name='anno_mask')
        
        self.W_polyak = cp.deepcopy(self.W) 
        self.c_polyak = cp.deepcopy(self.c)
        self.G_polyak = cp.deepcopy(self.G)
        
            
    def __deepcopy__(self,memo): 
        print "Warning: the deepcopy only copies the parameters, you SHOULD call compile_function for the functions"
        newone = type(self)()
        memo[id(self)] = newone
        old_dict = dict(self.__dict__)
        for key,val in old_dict.items():
            if key in ['train','valid','test']:
                print 'escape %s'%(key)
                pass
            else:
                newone.__dict__[key] = cp.deepcopy(val, memo)
        return newone  
    
      
    def build_graph(self, debug, hist_visual, hist_anno, global_feature, target,n_layer_to_build, W, c, V, b, G, U, dd, flag_train):
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
            
        
        hist_anno_dense = hist_anno.toarray()
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)
        if debug==True:
            mask_unif = 0.5*T.ones(shape=hist.shape, dtype=theano.config.floatX)
            
        else:
            mask_unif = 1.0 - self.rng_theano.uniform(size=hist.shape, low=0., high=1., dtype=theano.config.floatX)
        mask_counts = mask_unif*(hist+1)
        
        input = T.floor(mask_counts)*self.anno_mask
        hist = hist*self.anno_mask
        d = input.sum(axis = 1)
        D = hist.sum(axis = 1)
        predict = hist - input
        condition_bias = T.dot(global_feature, G)
        
        if self.preprocess_method == 'None':
            tmp_input = input
        elif self.preprocess_method == 'std':
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
            
            
        else:
            print 'Unknow preprocess method'
            exit(-1)
            
        
        first_tmp_input = tmp_input 
        if self.sup_option == 'full':
            if self.preprocess_method == 'None':
                tmp_sup_input = hist
            elif self.preprocess_method == 'std':
                std_full = T.std(hist, axis=1)
                tmp_sup_input = hist/(std_full[:, np.newaxis]+1e-16)
            elif self.preprocess_method == 'SPM':
                div_number = T.sqrt((hist**2).sum(axis=1))
                tmp_sup_input = input/(div_number[:,np.newaxis]+1e-16)
            else:
                print 'Unknow preprocess method'
                exit(-1)
            
        for i in xrange(n_layer_to_build):
            if i==0:
                dropout_mask = ifelse(T.neq(flag_train, 0) ,self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX), (1.-self.dropout_rate[i])*T.ones((tmp_input.shape[0],W[i].shape[1]), theano.config.floatX))
                h = self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*dropout_mask
                if self.sup_option == 'full':
                    h_sup = self.activation(T.dot(tmp_sup_input, W[i])+c[i]+condition_bias)*dropout_mask
            else:
                dropout_mask = ifelse(T.neq(flag_train, 0) ,self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX), (1.-self.dropout_rate[i])*T.ones((tmp_input.shape[0],W[i].shape[1]), theano.config.floatX))
                h = self.activation(T.dot(tmp_input, W[i])+c[i])*dropout_mask*1.0
                if self.sup_option == 'full':
                    h_sup = self.activation(T.dot(tmp_sup_input, W[i])+c[i])*dropout_mask*1.0
            tmp_input = h
            if self.sup_option == 'full':
                tmp_sup_input = h_sup
                #         G_value = (2*self.rng.rand(self.global_feat_size,self.hidden_size[0])-1)/(np.max([self.global_feat_size, self.hidden_size[0]]))
        log_prob_each_word = T.log(T.nnet.softmax_with_bias(T.dot(h, V),b)+1e-16)
        log_prob_each_bin = log_prob_each_word*predict
        nll = -log_prob_each_bin.sum(axis=1)
        
            
        #=====================sup_cost===============================
        if self.sup_option == 'full':
            prob_target = T.nnet.sigmoid(T.dot(h_sup, U)+dd)
        elif self.sup_option == 'partial':
            prob_target = T.nnet.sigmoid(T.dot(h, U)+dd)
        else:
            print "unknown supvervised option"
            exit(-1)
        cross_entropy = T.nnet.binary_crossentropy(prob_target, target).sum(axis=1)# the better the smaller (theano crossentropy add a minus here
        if self.normalize_by_document_size:
            unsup_cost = 1.0/(D-d)*nll
        else:
            unsup_cost = D/(D-d)/self.aver_words_count*nll
            
        cost = T.mean(unsup_cost*self.unsup_weight + cross_entropy)
#         T.mean(D/(D-d)*nll*self.unsup_weight + cross_entropy)
        log_prob_target = T.log(prob_target)
        return cost,log_prob_target,h, unsup_cost, cross_entropy, first_tmp_input, h_sup
    
    def build_compute_representation_graph(self, hist_visual, hist_anno, global_feature,n_layer_to_build, W, c,G, U, d, flag_train):
        
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
             
         
        hist_anno_dense = hist_anno.toarray()
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)




        condition_bias = T.dot(global_feature, G)
#           
        if self.preprocess_method == 'None':
            input = hist*self.anno_mask
            tmp_input = input
        elif self.preprocess_method == 'std':
            input = hist*self.anno_mask
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            input = hist*self.anno_mask
            
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
        else:
            print 'Unknow preprocess method'
            exit(-1)
            
            
        for i in xrange(n_layer_to_build):
            if i==0:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)
            else:
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i])
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i])*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i])*(1.-self.dropout_rate[i]))
            tmp_input = h
        log_prob_target = T.log(T.nnet.sigmoid(T.dot(h, U)+d))     
         
        return h,log_prob_target
    
    
    
    def compile_function(self, n_layers, trainset, validset):
        
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_feature')
        target = T.matrix(name='target')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
        cost,log_prob_target, hidden_representation, unsup_cost, cross_entropy, first_tmp_input, h_sup = self.build_graph(False, hist_visual, hist_anno, global_feature, target, n_layers, self.W, self.c, self.V, self.b, self.G,self.U, self.d, flag_train)
        
        params = [self.V, self.b, self.G, self.U, self.d]
        params.extend(self.W[:n_layers])
        params.extend(self.c[:n_layers])
        
        polyaks = [self.V_polyak, self.b_polyak, self.G_polyak, self.U_polyak, self.d_polyak]
        polyaks.extend(self.W_polyak[:n_layers])
        polyaks.extend(self.c_polyak[:n_layers])
        
        params_gradient = [T.grad(cost, param) for param in params]
        
        
        updates = []

        for param, param_gradient, polyak in zip(params, params_gradient, polyaks):
            param_updated = param - self.dec_learning_rate*param_gradient
            if param.get_value(borrow=True).ndim==2:
                col_norms = T.sqrt(T.sum(T.sqr(param_updated), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.length_limit))
                col_scale = desired_norms / (1e-16 + col_norms)
                updates.append((param, param_updated*col_scale))
            else:
                updates.append((param, param_updated))
                
            polyak_updated = self.polyakexp_weight*polyak + (1-self.polyakexp_weight)* param_updated      
            updates.append((polyak, polyak_updated))
            
            
        updates.append((self.dec_learning_rate, self.dec_learning_rate*self.decrease_constant))
            
        self.train = theano.function(inputs = [index],
                                    updates = updates,
#                                      outputs = [cost, log_prob_target, unsup_cost, cross_entropy, hidden_representation, first_tmp_input, h_sup],  
                                    outputs = [cost, log_prob_target, unsup_cost, cross_entropy],  
                                     givens = {
                                               hist_visual:trainset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:trainset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:trainset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               target:trainset['targets'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(1,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )   
#         theano.printing.pydotprint(self.train, outfile='/home/local/USHERBROOKE/zhey2402/DeepDocNADE/pic.png')
        self.valid = theano.function(inputs = [index],
#                                      updates = updates,
#                                     outputs = [cost, log_prob_target, unsup_cost, cross_entropy, hidden_representation, first_tmp_input, h_sup], 
                                    outputs = [cost, log_prob_target, unsup_cost, cross_entropy], 
                                     givens = {
                                               hist_visual:validset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:validset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:validset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               target:validset['targets'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(0,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )
    def compile_LayerByLayer_function(self, n_layers, trainset, validset):
        
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.cast(T.scalar('index'), 'int32')
        flag_train = T.scalar(name='flag_train')
        cost,hidden_representation = self.build_unsupervised_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G, flag_train)
        
        params = [self.V, self.b, self.G, self.W[n_layers-1], self.c[n_layers-1]]
        polyaks = [self.V_polyak, self.b_polyak, self.G_polyak, self.W_polyak[n_layers-1], self.c_polyak[n_layers-1]]

        
        params_gradient = [T.grad(cost, param) for param in params]
        
        
        updates = []

        for param, param_gradient, polyak in zip(params, params_gradient, polyaks):
            param_updated = param - self.dec_learning_rate*param_gradient
            if param.get_value(borrow=True).ndim==2:
                col_norms = T.sqrt(T.sum(T.sqr(param_updated), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.length_limit))
                col_scale = desired_norms / (1e-16 + col_norms)
                updates.append((param, param_updated*col_scale))
            else:
                updates.append((param, param_updated))
                
            polyak_updated = self.polyakexp_weight*polyak + (1-self.polyakexp_weight)* param_updated      
            updates.append((polyak, polyak_updated))
            
            
        updates.append((self.dec_learning_rate, self.dec_learning_rate*self.decrease_constant))
            
        self.train = theano.function(inputs = [index],
                                    updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:trainset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:trainset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:trainset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(1,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )   
#         theano.printing.pydotprint(self.train, outfile='/home/local/USHERBROOKE/zhey2402/DeepDocNADE/pic.png')
        self.valid = theano.function(inputs = [index],
#                                      updates = updates,
                                     outputs = cost,  
                                     givens = {
                                               hist_visual:validset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:validset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:validset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(0,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )  
        
    def compile_compute_representation_function(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_feature')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
#         cost,hidden_representation,input, anno_weighted_mask = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G)
        hidden_representation, log_prob_target = self.build_compute_representation_graph(hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.G, self.U, self.d, flag_train)
        self.compute_representation = theano.function(inputs = [index],
                                                      outputs = [hidden_representation,log_prob_target],  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
#                                                       mode='DebugMode'
                                                     ) 
    def compile_compute_representation_function_polyak(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_feature')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
#         cost,hidden_representation,input, anno_weighted_mask = self.build_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G)
        hidden_representation, log_prob_target = self.build_compute_representation_graph(hist_visual, hist_anno, global_feature, n_layers, self.W_polyak, self.c_polyak, self.G_polyak, self.U_polyak, self.d_polyak, flag_train)
        self.compute_representation = theano.function(inputs = [index],
                                                      outputs = [hidden_representation,log_prob_target],  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
#                                                       mode='DebugMode'
                                                     )
    def build_unsupervised_graph(self, debug, hist_visual, hist_anno, global_feature, n_layer_to_build, W, c, V, b, G, flag_train):
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
            
        
        hist_anno_dense = hist_anno.toarray()
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)
        if debug==True:
            mask_unif = 0.5*T.ones(shape=hist.shape, dtype=theano.config.floatX)
            
        else:
            mask_unif = 1.0 - self.rng_theano.uniform(size=hist.shape, low=0., high=1., dtype=theano.config.floatX)
        mask_counts = mask_unif*(hist+1)
        input = T.floor(mask_counts)*self.anno_mask
        hist = hist*self.anno_mask
        d = input.sum(axis = 1)
        D = hist.sum(axis = 1)
        predict = hist - input
        condition_bias = T.dot(global_feature, G)
        
        if self.preprocess_method == 'None':
            tmp_input = input
        elif self.preprocess_method == 'std':
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
            
        else:
            print 'Unknow preprocess method'
            exit(-1)
        
#         tmp_input = input
        for i in xrange(n_layer_to_build):
            if i==0:
                
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)
            else:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i])*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i])*(1.-self.dropout_rate[i]))
#                 h = self.activation(T.dot(tmp_input, W[i])+c[i])
            tmp_input = h
        log_prob_each_word = T.log(T.nnet.softmax_with_bias(T.dot(h, V),b)+1e-16)
        log_prob_each_bin = log_prob_each_word*predict
        nll = -log_prob_each_bin.sum(axis=1)
        #TODO:??divide D??
        if self.normalize_by_document_size:
            cost = T.mean(1.0/(D-d)*nll)
        else:
            cost = T.mean(D/(D-d)/self.aver_words_count*nll)
        
        return cost,h
    
    
    def compile_unsupervised_function(self, n_layers, trainset, validset):
        
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_features')
        index = T.cast(T.scalar('index'), 'int32')
        flag_train = T.scalar(name='flag_train')
        cost,hidden_representation = self.build_unsupervised_graph(False, hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G, flag_train)
        
        params = [self.V, self.b, self.G]
        params.extend(self.W[:n_layers])
        params.extend(self.c[:n_layers])
        
        polyaks = [self.V_polyak, self.b_polyak, self.G_polyak]
        polyaks.extend(self.W_polyak[:n_layers])
        polyaks.extend(self.c_polyak[:n_layers])
        
        params_gradient = [T.grad(cost, param) for param in params]
        
        
        updates = []

        for param, param_gradient, polyak in zip(params, params_gradient, polyaks):
            param_updated = param - self.dec_learning_rate*param_gradient
            if param.get_value(borrow=True).ndim==2:
                col_norms = T.sqrt(T.sum(T.sqr(param_updated), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.length_limit))
                col_scale = desired_norms / (1e-16 + col_norms)
                updates.append((param, param_updated*col_scale))
            else:
                updates.append((param, param_updated))
                
            polyak_updated = self.polyakexp_weight*polyak + (1-self.polyakexp_weight)* param_updated      
            updates.append((polyak, polyak_updated))
            
            
        updates.append((self.dec_learning_rate, self.dec_learning_rate*self.decrease_constant))
            
        self.train = theano.function(inputs = [index],
                                    updates = updates,
                                     outputs = [cost,hidden_representation],  
                                     givens = {
                                               hist_visual:trainset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:trainset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:trainset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(1,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )   
#         theano.printing.pydotprint(self.train, outfile='/home/local/USHERBROOKE/zhey2402/DeepDocNADE/pic.png')
        self.valid = theano.function(inputs = [index],
#                                      updates = updates,
                                     outputs = [cost,hidden_representation],  
                                     givens = {
                                               hist_visual:validset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               hist_anno:validset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                               global_feature:validset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                               flag_train:np.asarray(0,dtype=theano.config.floatX)
                                               },
#                                      mode='DebugMode'
                                    )
    
    
    def build_graph_generateTEXT(self, hist_visual, hist_anno, global_feature,n_layer_to_build, W, c, V, b, G, flag_train):
        
        if n_layer_to_build <1:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if n_layer_to_build > self.n_layers:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
            
        
        hist_anno_dense = hist_anno.toarray()*0.0
        hist = T.concatenate([hist_visual, hist_anno_dense], axis=1)
        hist = hist*self.anno_mask
        
        condition_bias = T.dot(global_feature, G)
#           
        if self.preprocess_method == 'None':
            input = hist*self.anno_mask
            tmp_input = input
        elif self.preprocess_method == 'std':
            input = hist*self.anno_mask
            std = T.std(input, axis=1)
            tmp_input = input/(std[:, np.newaxis]+1e-16)
        elif self.preprocess_method == 'SPM':
            input = hist*self.anno_mask
            div_number = T.sqrt((input**2).sum(axis=1))
            tmp_input = input/(div_number[:,np.newaxis]+1e-16)
        else:
            print 'Unknow preprocess method'
            exit(-1)
            
            
        for i in xrange(n_layer_to_build):
            if i==0:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i]+condition_bias)*(1.-self.dropout_rate[i]))
            else:
                h = ifelse(T.neq(flag_train, 0) ,self.activation(T.dot(tmp_input, W[i])+c[i])*(self.rng_theano.binomial(n=1, p=1.0-self.dropout_rate[i], size = (tmp_input.shape[0],W[i].shape[1]),dtype=theano.config.floatX)), self.activation(T.dot(tmp_input, W[i])+c[i])*(1.-self.dropout_rate[i]))
            tmp_input = h
         
            
        log_prob_each_word = T.log(T.nnet.softmax_with_bias(T.dot(h, V),b)+1e-16)
        
            
        return h, log_prob_each_word 
    
    def compile_generateTEXT_function(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_feature')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
        hidden_representation, log_prob_each_word = self.build_graph_generateTEXT(hist_visual, hist_anno, global_feature, n_layers, self.W, self.c, self.V, self.b, self.G, flag_train)
        self.generateTEXT = theano.function(inputs = [index],
                                                      outputs = [hidden_representation,log_prob_each_word],  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
                                                     ) 
    def compile_generateTEXT_function_polyak(self,n_layers, dataset): 
        hist_visual = T.matrix(name='hist_visual')
        hist_anno = S.csr_matrix(name='hist_anno')
        global_feature = T.matrix(name='global_feature')
        index = T.lscalar('index')
        flag_train = T.scalar(name='flag_train')
        hidden_representation, log_prob_each_word = self.build_graph_generateTEXT(hist_visual, hist_anno, global_feature, n_layers, self.W_polyak, self.c_polyak, self.V_polyak, self.b_polyak, self.G_polyak, flag_train)
        self.generateTEXT = theano.function(inputs = [index],
                                                      outputs = [hidden_representation,log_prob_each_word],  
                                                      givens = {
                                                                hist_visual:dataset['hists_visual'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                hist_anno:dataset['hists_anno'][index*self.batch_size:(index+1)*self.batch_size,:],
                                                                global_feature:dataset['global_features'][index*self.batch_size:(index+1)*self.batch_size, :],
                                                                
                                                                flag_train:np.asarray(0,dtype=theano.config.floatX)
                                                               }
                                                     )
    
    def verify_gradients(self):
        
        def fun(W0,W1, c0, c1, V, b,G):
            
            hist_visual = np.array([[1,2,3,4,5,6,7,8,9,0]], dtype = theano.config.floatX)
            hist_anno = sp.csr_matrix([[0,0,1,0,0,0,1,2,0,0]], dtype = theano.config.floatX)
            global_feature = np.array([[1,2,3,4,5,6,7,8,9,0]], dtype = theano.config.floatX)
            n_layers = 2
            cost, h=self.build_graph(True, hist_visual, hist_anno, global_feature, n_layers, 
                                  [W0,W1], [c0,c1], V, b, G)
                
            return cost
        print 'Warning: verify_gradient will reinitialize the model!!!'
        self.hidden_size = [100,100]
        self.n_classes = 7
        self.dropout_rate = 0.5
        self.activation = activation_functions['reclin']
        self.n_layers = len(self.hidden_size)
        self.initialize(10,10,10)
        rng = np.random.RandomState(42)
        
        
#         rng = np.random.RandomState(42)
        theano.tests.unittest_tools.verify_grad(fun, [self.W[0].get_value(), self.W[1].get_value(),self.c[0].get_value(), self.c[1].get_value(),
                                                      self.V.get_value(), self.b.get_value(), self.G.get_value()], rng = rng)
        
        
        
        
    def remove_activation(self):
        
        del self.activation
        
    def add_activation(self):
        
        self.activation = activation_functions[self.activation_function_name]
        
    def remove_supervised_layer(self):
        
        if hasattr(self, 'U'):
            del self.U
        if hasattr(self, 'd'):
            del self.d
        
    def add_supervised_layer(self, layer_id):
        
        if layer_id <0:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if layer_id > self.n_layers-1:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
#         U_value = 1*(2*self.rng.rand(self.hidden_size[layer_id] ,self.n_classes)-1)/(np.max([self.hidden_size[layer_id],self.n_classes]))
#         U_value = self.rng.uniform(-np.sqrt(0.05)/(self.hidden_size[layer_id]+self.n_classes), np.sqrt(0.05)/(self.hidden_size[layer_id]+self.n_classes), size=(self.hidden_size[layer_id],self.n_classes))
        U_value = (1.0**(layer_id))*self.rng.uniform(-np.sqrt(6)/np.sqrt(self.hidden_size[layer_id]+self.n_classes), np.sqrt(6)/np.sqrt(self.hidden_size[layer_id]+self.n_classes), size=(self.hidden_size[layer_id],self.n_classes))
#         U_value = 0.001*generate_SparseConnectionMat(self.rng, self.hidden_size[layer_id],self.n_classes, self.n_connection, self.rescale, self.bias)
        U_value = np.asarray(U_value, theano.config.floatX)
        d_value = np.zeros((self.n_classes), theano.config.floatX)   
        self.U = theano.shared(value=U_value, name='U') 
        self.d = theano.shared(value=d_value, name='d') 
        
        self.U_polyak = cp.deepcopy(self.U)
        self.d_polyak = cp.deepcopy(self.d)   
    
    def remove_top_layer(self):
        if hasattr(self, 'V'):
            del self.V
        if hasattr(self, 'b'):
            del self.b
    
    def add_top_layer(self, layer_id):
        '''
        layer_id is the id of the hidden layer (starting from 0) on which we build the top layer to compute the conditionals
        '''
        if layer_id <0:
            print 'there is at least 1 hidden layer'
            exit(-1)
        if layer_id > self.n_layers-1:
            print 'exceed the max number of hidden layers'
            print 'the max number of hidden layers is %d'%(self.n_layers)
            exit(-1)
#         V_value = 1*(2*self.rng.rand(self.hidden_size[layer_id],self.voc_size+self.anno_voc_size)-1)/(np.max([self.voc_size+self.anno_voc_size, self.hidden_size[layer_id]])) 
#         V_value = self.rng.uniform(-np.sqrt(0.05)/(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), np.sqrt(0.05)/(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), size=( self.hidden_size[layer_id],self.voc_size+self.anno_voc_size))
        V_value = self.rng.uniform(-np.sqrt(6)/np.sqrt(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), np.sqrt(6)/np.sqrt(self.voc_size+self.anno_voc_size + self.hidden_size[layer_id]), size=( self.hidden_size[layer_id],self.voc_size+self.anno_voc_size))
#         V_value = 0.01*generate_SparseConnectionMat(self.rng, self.hidden_size[layer_id],self.voc_size+self.anno_voc_size, self.n_connection, self.rescale, self.bias)
        V_value = np.asarray(V_value, theano.config.floatX)
        self.V = theano.shared(value = V_value, name = 'V')
        b_value = np.zeros((self.voc_size+self.anno_voc_size), theano.config.floatX)
        self.b = theano.shared(value = b_value, name = 'b')    
        
        self.V_polyak = cp.deepcopy(self.V)
        self.b_polyak = cp.deepcopy(self.b)
        
    def copy_parameters(self, source):
        
        self.V.set_value(source.V.get_value())
        self.b.set_value(source.b.get_value())
        self.V_polyak.set_value(source.V_polyak.get_value())
        self.b_polyak.set_value(source.b_polyak.get_value())
        for i in xrange(self.n_layers):
            self.W[i].set_value(source.W[i].get_value())
            self.c[i].set_value(source.c[i].get_value())
            self.W_polyak[i].set_value(source.W_polyak[i].get_value())
            self.c_polyak[i].set_value(source.c_polyak[i].get_value())
        self.G.set_value(source.G.get_value())
        self.G_polyak.set_value(source.G_polyak.get_value())
        self.dec_learning_rate.set_value(source.dec_learning_rate.get_value())
        
        if hasattr(source, 'U'):
            self.U.set_value(source.U.get_value())
            self.U_polyak.set_value(source.U_polyak.get_value())
        if hasattr(source,'d'):
            self.d.set_value(source.d.get_value())
            self.d_polyak.set_value(source.d_polyak.get_value())
