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

@author: Yin Zheng
'''
import MIR_Flickr_Theano_Unlab as MFU
from SupDeepDocNADE import DeepDocNADE
import copy
import numpy as np
from itertools import izip 
import random
import collections
import cPickle
import sys, os
import fcntl
import time as t
import theano
import theano.tensor as T
import theano.sparse as S
from liblinearutil import *
import gc
import shutil
import glob

# activation_functions = {"sigmoid": theano.tensor.nnet.sigmoid, "reclin": lambda x: x * (x > 0), "tanh": theano.tensor.tanh}
def get_done_text(start_time):
    sys.stdout.flush()
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)
sys.argv.pop(0);    # Remove first argument

# Check if all options are provided
if 11 != len(sys.argv):
    print "Usage: python run_pretrain.py n_pretrain pre_learning_rate hidden_size activation_function  dropout_rate model_file_dir unlab_dataset_dir batch_size anno_weight platform   polyakexp_weight model_init"
    sys.exit()

#     scene15.obtain(r'/home/ian/ml_datasets/Scene15')
str2bool = {'True':True, 'False': False}
n_pretrain = int(sys.argv[0])
pre_learning_rate = float(sys.argv[1])
hidden_size_split = (sys.argv[2]).split('_')
hidden_size =  [int(x) for x in hidden_size_split]
activation_function = sys.argv[3]
dropout_split = (sys.argv[4]).split('_')
dropout_rate =  [float(x) for x in dropout_split]
model_file_dir = sys.argv[5]
unlab_dataset_dir = sys.argv[6]
batch_size = int(sys.argv[7])
normalize_by_document_size = False
anno_weight = float(sys.argv[8])
log_option = "NoLog"
spatial_pyramid =  [1]
platform = 'PC'
scaled_method = 'std'
length_limit = float(100.0)
decrease_constant = float(1.0)
polyakexp_weight = float(sys.argv[9])
pretrained_models_dir = sys.argv[10]

if not os.path.exists(unlab_dataset_dir):
    print 'no such file for dataset'
    exit(-1)
    
    
#================================================= search the potential model that match the option========================================
if normalize_by_document_size:
    template_ID = 'Wholelayers__%s__%s__%s__%s__*__%s__%s__%f__%s__%f__%f__%f__normalized_by_doc_size'%(sys.argv[2],activation_function, log_option, sys.argv[11], pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight)
else:
    template_ID = 'Wholelayers__%s__%s__%s__%s__*__%s__%s__%f__%s__%f__%f__%f'%(sys.argv[2],activation_function, log_option, sys.argv[11], pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight )
template_model_name = os.path.join(pretrained_models_dir, template_ID+'___model.pkl')

potential_model_name = glob.glob(template_model_name)

init_model = 'None'
n_trained = -np.inf
for tmp_model in potential_model_name:
    tmp_config = tmp_model.split('__')
    tmp_n_trained = int(tmp_config[5])
    if tmp_n_trained > n_trained:
        n_trained = tmp_n_trained
        init_model = tmp_model
        
#===========================================================================================================================================
if not os.path.exists(init_model):
    print 'no init model found, we will train the model from epoch 0'
    flag_continue = False
else:
    print 'the model we based on is %s'%(init_model)
    flag_continue = True
#     exit(-1)    


if platform == 'Guilinmin':
    src = unlab_dataset_dir
    dst = os.path.join('/dev/shm/', os.environ['PBS_JOBID'])
#     dst = os.path.join('/home/local/USHERBROOKE/zhey2402/localscratch', os.environ['PBS_JOBID'])
    print src
    print dst
    
    print 'starting copying'
    for i in xrange(1, 51):
        src_filename = os.path.join(src, 'unlabeled'+str(i)+'.npz.gz')
        start_copy = t.time()
        shutil.copy(src_filename, dst)
        print 'Copy file_ID %d to shm'%(i), get_done_text(start_copy)
    
    
    start_copy = t.time()
    src_filename = os.path.join(src, 'others.tar.gz')
    shutil.copy(src_filename, dst)
    print 'Copy others.tar.gz to shm', get_done_text(start_copy)
    
    start_extract = t.time()
    path_tarfile = os.path.join(dst, 'others.tar.gz')
    cmd = 'tar xvfzm '+path_tarfile + ' -C ' + dst
    print cmd
    os.system(cmd)
    print 'Extract other.tar.gz', get_done_text(start_extract)
    
    start_remove_tar = t.time()
    cmd = 'rm ' + path_tarfile
    print cmd
    print 'Removing other.tar.gz file to save space'
    os.system(cmd)
    print 'Remove tar', get_done_text(start_remove_tar)
    
    unlab_dataset_dir = dst
    


if flag_continue:
    init_config = init_model.split('__')
    init_hidden_size = init_config[1]
    init_activation = init_config[2]
    init_logoption = init_config[3]
    init_spatial = init_config[4]
    init_epoch = int(init_config[5])
    init_lr = init_config[6]
    init_dropout = init_config[7]
    init_annoweight = float(init_config[8])
    init_scale = init_config[9]
    init_lengthlimit = float(init_config[10])
    init_decreaseconst = float(init_config[11])
    init_polyweight = float(init_config[12]) 
    
    init_model = os.path.expanduser(init_model)
    if os.path.isfile(init_model):
        model_file = open(init_model, 'rb')
        model_init = cPickle.load(model_file)
        model_file.close()
        model_init.add_activation()
    else:
        print 'ERROR: init model not found'
        exit(-1)
    
    assert(model_init.hidden_size == hidden_size)
    assert(model_init.learning_rate == pre_learning_rate)
    assert(model_init.activation_function_name == activation_function)
    assert(model_init.dropout_rate == dropout_rate)
    assert(model_init.normalize_by_document_size == normalize_by_document_size)
    assert(model_init.anno_weight == anno_weight)
    assert(model_init.batch_size == batch_size)
    assert(model_init.preprocess_method == scaled_method)
    assert(model_init.length_limit == length_limit)
    assert(model_init.decrease_constant == decrease_constant)
    assert(model_init.polyakexp_weight == polyakexp_weight)
    assert(init_logoption == log_option)
    assert(init_spatial == '1') 
    if init_epoch >= n_pretrain:
        print 'the model is trained %d epoches, which equals or exceeds the number %d you required'%(init_epoch, n_pretrain)
        exit(-1)
else:
    init_epoch = 0 
    
model = DeepDocNADE(hidden_size = hidden_size,
                    learning_rate = pre_learning_rate,
                    activation_function = activation_function,
                    word_representation_size = 0,
                    dropout_rate = dropout_rate,
                    normalize_by_document_size = normalize_by_document_size,
                    anno_weight = anno_weight,
                    batch_size = batch_size,
                    preprocess_method = scaled_method,
                    length_limit = length_limit,
                    decrease_constant = decrease_constant,
                    polyakexp_weight = polyakexp_weight,
                    seed_np = init_epoch + 1126,
                    seed_theano = init_epoch + 1959
                    )        
initialized = False
flag_compiled = False    
    
#================================================= create a object used to save model==============================
copy_model = copy.deepcopy(model)
# copy_model.remove_activation()
train_ahead = n_pretrain   
print 'begin pretrain using unlabeled data...'
print 'we need to train it %d more epoches'%(min(init_epoch+train_ahead,n_pretrain)-init_epoch)


n_layers = model.n_layers
# for n_build in xrange(n_layers):
n_build = n_layers-1
epoch = init_epoch

print '\n### Training DeepDocNADE using unlabeled data, n_layers=%d ###'%(n_build+1)
start_training_time = t.time()
model.dec_learning_rate.set_value(model.learning_rate)
copy_model.dec_learning_rate.set_value(copy_model.learning_rate)
if initialized:
    model.remove_top_layer()
    model.add_top_layer(n_build)
while(epoch < min(init_epoch+train_ahead,n_pretrain)):
    
    epoch += 1
    print 'Epoch {0}'.format(epoch)
    start_time_epoch = t.time()
    cost_train = []
    for file_id in xrange(1,51):
        
        start_time = t.time()
        start_time_loaddata = t.time()
        #===================extract corresponding unlabeled(file_id).npz.gz file=================================
        if platform == 'Guilinmin':
            start_extract = t.time()
            path_tarfile = os.path.join(unlab_dataset_dir, 'unlabeled'+str(file_id)+'.npz.gz')
            cmd = 'tar xvfzm '+path_tarfile + ' -C ' + unlab_dataset_dir
            print cmd
            os.system(cmd)
            print 'Extract file_ID %d'%(file_id), get_done_text(start_extract)
        #===============================LOAD file==================================================
        unlabel_raw = MFU.load(unlab_dataset_dir, file_id, log_option, spatial_pyramid)
        
        #======================================remove unlabeled(file_id).npz.gz============================
        if platform == 'Guilinmin':
            start_remove_tar = t.time()
            path_npzfile = os.path.join(unlab_dataset_dir, 'unlabeled'+str(file_id)+'.npz')
            cmd = 'rm ' + path_npzfile
            print cmd
            os.system(cmd)
            print 'Remove file_ID %d'%(file_id), get_done_text(start_remove_tar)
        #==================================================================================================
        print '\tTraining   ...',
        sys.stdout.write("Load data cost {:.4f} seconds    ".format(t.time() - start_time_loaddata))
        if not flag_compiled:
            unlabel = {}
            unlabel['hists_visual'] = theano.shared(np.asarray(unlabel_raw['hists_visual'], theano.config.floatX), borrow=False)
            unlabel['hists_anno'] = theano.shared(unlabel_raw['hists_anno'].astype(theano.config.floatX), borrow=False)
            unlabel['global_features'] = theano.shared(np.asarray(unlabel_raw['global_features'], theano.config.floatX), borrow=False)
        else:
            unlabel['hists_visual'].set_value(np.asarray(unlabel_raw['hists_visual'], theano.config.floatX))
            unlabel['hists_anno'].set_value(unlabel_raw['hists_anno'].astype(theano.config.floatX))
            unlabel['global_features'].set_value(np.asarray(unlabel_raw['global_features'], theano.config.floatX))
            
        n_train_batches = unlabel_raw['meta']['length']/batch_size
        
        aver_words_count = unlabel_raw['hists_visual'].sum(axis=1).mean()
        sys.stdout.write("aver word counts is {:.4f} ".format(aver_words_count))            
        if not initialized:
            spatial_split = np.asarray(spatial_pyramid, np.int32)**2*unlabel_raw['meta']['voc_size']
            region_split = np.append(spatial_split, unlabel_raw['meta']['text_voc_size'])
            region_split = np.add.accumulate(region_split)
            
            
            model.initialize(unlabel_raw['meta']['voc_size']*unlabel_raw['meta']['n_regions'], unlabel_raw['meta']['text_voc_size'], unlabel_raw['meta']['global_feat_size'], region_split)
            model.remove_top_layer()
            model.add_top_layer(n_build)
            copy_model.initialize(unlabel_raw['meta']['voc_size']*unlabel_raw['meta']['n_regions'], unlabel_raw['meta']['text_voc_size'], unlabel_raw['meta']['global_feat_size'], region_split)
            copy_model.remove_top_layer()
            copy_model.add_top_layer(n_build)
            del copy_model.rng_theano
            del copy_model.rng
            if flag_continue:
                model.copy_parameters(model_init)
                copy_model.copy_parameters(model_init)
                del model_init
            initialized = True
        model.aver_words_count = aver_words_count
        copy_model.aver_words_count = aver_words_count
        
        start_time_process = t.time()  
        if not flag_compiled: 
            model.compile_function(n_build+1, unlabel, unlabel)
            flag_compiled = True
        for minibatch_index in range(n_train_batches):
            cost_value = model.train(minibatch_index)
            cost_train += [cost_value]
        sys.stdout.write("Process data cost {:.4f} seconds    ".format(t.time() - start_time_process))
        del unlabel_raw
#         del model.train
#         del model.valid
#         del unlabel
        gc.collect()
        
        print 'Train     :', 'File ID %d'%(file_id), get_done_text(start_time)
#         unlabel.clear()
    train_cost_error = np.asarray(cost_train).mean()
    print '\tTraining   ...',
    print 'Train     :', " Cost Error: {0:.6f}".format(train_cost_error), get_done_text(start_time_epoch)
#     if np.mod(epoch,2)==0:
#         copy_model.copy_parameters(model)
#         del model
#         gc.collect()
#         model = copy.deepcopy(copy_model)
    if np.mod(epoch, 25)==0:
        if normalize_by_document_size:
            cPickle_ID = 'Wholelayers__%s__%s__%s__%s__%d__%s__%s__%f__%s__%f__%f__%f__normalized_by_doc_size'%(sys.argv[2],activation_function, log_option, sys.argv[11], epoch, pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight)
        else:
            cPickle_ID = 'Wholelayers__%s__%s__%s__%s__%d__%s__%s__%f__%s__%f__%f__%f'%(sys.argv[2],activation_function, log_option, sys.argv[11], epoch, pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight )
        cPickle_model_name = os.path.join(model_file_dir, cPickle_ID+'___model.pkl')
        copy_model.copy_parameters(model)
        copy_model.remove_activation()
        
        saved_model_list = open(os.path.join(model_file_dir, 'saved_model_list.txt'), 'a')
        fcntl.flock(saved_model_list.fileno(), fcntl.LOCK_EX)
        model_file = open(cPickle_model_name, 'wb')
        cPickle.dump(copy_model, model_file,protocol=cPickle.HIGHEST_PROTOCOL)
        model_file.close()
        saved_model_list.write(cPickle_model_name+'\n')
        saved_model_list.close() # unlocks the file
        copy_model.add_activation()
        print cPickle_model_name
        print 'is saved'
        
print '\n### Pre_Training, n_layers=%d'%(n_build+1), get_done_text(start_training_time)
# copy_model.copy_parameters(model)
# del model
# gc.collect()
# model = copy.deepcopy(copy_model)
    
if normalize_by_document_size:
    cPickle_ID = 'Wholelayers__%s__%s__%s__%s__%d__%s__%s__%f__%s__%f__%f__%f__normalized_by_doc_size'%(sys.argv[2],activation_function, log_option, sys.argv[11], min(init_epoch+train_ahead,n_pretrain), pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight)
else:
    cPickle_ID = 'Wholelayers__%s__%s__%s__%s__%d__%s__%s__%f__%s__%f__%f__%f'%(sys.argv[2],activation_function, log_option, sys.argv[11], min(init_epoch+train_ahead,n_pretrain), pre_learning_rate, sys.argv[4], anno_weight, scaled_method, length_limit, decrease_constant, polyakexp_weight)
cPickle_model_name = os.path.join(model_file_dir, cPickle_ID+'___model.pkl')    

# copy_model.copy_parameters(model)
model.remove_activation()
del model.train
del model.valid
del unlabel
del model.rng_theano
del model.rng
gc.collect()

saved_model_list = open(os.path.join(model_file_dir, 'saved_model_list.txt'), 'a')
fcntl.flock(saved_model_list.fileno(), fcntl.LOCK_EX)
model_file = open(cPickle_model_name, 'wb')
cPickle.dump(model, model_file,protocol=cPickle.HIGHEST_PROTOCOL)
model_file.close()
saved_model_list.write(cPickle_model_name+'\n')
saved_model_list.close() # unlocks the file


print cPickle_model_name
print 'is saved'