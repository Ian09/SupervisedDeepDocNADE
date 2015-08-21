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
'''

import MIR_Flickr_Theano_lab as MF
from SupDeepDocNADE import SupDeepDocNADE
import copy
import numpy as np
from itertools import izip 
import random
import collections
# import cPickle
import sys, os
import fcntl
import time as t
import theano
import theano.tensor as T
import theano.sparse as S
import cPickle
from sklearn.metrics import average_precision_score
from liblinearutil import *
import gc
sys.argv.pop(0);    # Remove first argument
def get_done_text(start_time):
    sys.stdout.flush()
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)

def softmax(a):
    max_a = np.amax(a , axis=1)
    max_a = max_a.reshape(max_a.shape[0], 1)
    e = np.exp(a-max_a)
    dist = e/np.sum(e, axis = 1)[:, np.newaxis]
    return dist

# activation_functions = {"sigmoid": theano.tensor.nnet.sigmoid, "reclin": lambda x: x * (x > 0), "tanh": theano.tensor.tanh}
# Check if all options are provided
if 17 != len(sys.argv):
    print "Usage: python run_SupDeepDocNADE.py folder_ID use_pretrain max_iter look_ahead hidden_size \
    learning_rate unsup_weight activation_function Linear_minC, Linear_maxC, \
    dropout_rate uniresult_dir Pretrain_model_name\
    lab_dataset_dir batch_size anno_weight\
     polyakexp_weight"
    sys.exit()

folder_ID = int(sys.argv[0])
use_pretrain = str2bool[sys.argv[1]]
max_iter = int(sys.argv[2])
look_ahead = int(sys.argv[3])
hidden_size_split = (sys.argv[4]).split('_')
hidden_size =  [int(x) for x in hidden_size_split]
learning_rate = float(sys.argv[5])
unsup_weight = float(sys.argv[6])
activation_function = sys.argv[7]
sup_option = 'full'
Linear_minC = float(sys.argv[8])
Linear_maxC = float(sys.argv[9])
dropout_split = (sys.argv[10]).split('_')
dropout_rate =  [float(x) for x in dropout_split]
uniresult_dir = sys.argv[11]
full_path_premodel = sys.argv[12]
lab_dataset_dir = sys.argv[13]
batch_size = int(sys.argv[14])
normalize_by_document_size = False
anno_weight = float(sys.argv[15])
log_option = "NoLog"
spatial_pyramid =  [1]
scaled_method = 'std'
length_limit = float(100.0)
decrease_constant = float(1.0)
polyakexp_weight = float(sys.argv[16])


        

def compute_AP_Prec50(preds, targets):
    
    targets_sorted = targets[(-preds.T).argsort().flatten()]
    cumsum = targets_sorted.cumsum()
    prec = cumsum / np.arange(1.0, 1 + targets.shape[0])
    ap = average_precision_score(targets, preds)
    prec50 = prec[50]
    return ap, prec50


def compute_MAP_Prec50(predictions, targets):
    
    numdims = predictions.shape[1]
    ap = 0
    prec50 = 0 
    ap_list = []
    prec50_list = []
    
    for i in range(numdims):
      this_ap, this_prec = compute_AP_Prec50(predictions[:,i], targets[:,i])
      ap_list.append(this_ap)
      prec50_list.append(this_prec)
      ap += this_ap
      prec50 += this_prec
    map = ap/numdims
    mprec50 = prec50/numdims
    return map, mprec50
        
        
    

str2bool = {'True':True, 'False': False}
folder_ID = int(sys.argv[0])
use_pretrain = str2bool[sys.argv[1]]
max_iter = int(sys.argv[2])
look_ahead = int(sys.argv[3])
hidden_size_split = (sys.argv[4]).split('_')
hidden_size =  [int(x) for x in hidden_size_split]
learning_rate = float(sys.argv[5])
unsup_weight = float(sys.argv[6])
activation_function = sys.argv[7]
sup_option = 'full'
Linear_minC = float(sys.argv[8])
Linear_maxC = float(sys.argv[9])
dropout_split = (sys.argv[10]).split('_')
dropout_rate =  [float(x) for x in dropout_split]
uniresult_dir = sys.argv[11]
full_path_premodel = sys.argv[12]
lab_dataset_dir = sys.argv[13]
batch_size = int(sys.argv[14])
normalize_by_document_size = False
anno_weight = float(sys.argv[15])
log_option = "NoLog"
spatial_pyramid =  [1]
scaled_method = 'std'
length_limit = float(100.0)
decrease_constant = float(1.0)
polyakexp_weight = float(sys.argv[16])

file_name_Linear = 'Polyak_Linear_Flickr_SupDeepDocNADE_%s__%s__%s.txt' %(sys.argv[0], activation_function, log_option)
uniresultfile_name_Linear = os.path.join(uniresult_dir, file_name_Linear)
print uniresultfile_name_Linear
rng_shuffle = np.random.mtrand.RandomState(1111)
if not os.path.exists(lab_dataset_dir):
    print 'label dataset not found'
    exit(-1)

    
    
print 'train using labeled data'  
    

dataset = MF.load(lab_dataset_dir, folder_ID, log_option, spatial_pyramid)
trainset_raw = dataset['train']
validset_raw = dataset['valid']
testset_raw = dataset['test']
n_classes = trainset_raw['meta']['n_classes']

train_labels = trainset_raw['targets']
valid_labels = validset_raw['targets']
test_labels = testset_raw['targets']

trainset = {}
validset = {}
testset = {}
trainset['hists_visual'] = theano.shared(np.asarray(trainset_raw['hists_visual'], theano.config.floatX))
trainset['hists_anno'] = theano.shared(trainset_raw['hists_anno'].astype(theano.config.floatX))
trainset['global_features'] = theano.shared(np.asarray(trainset_raw['global_features'], theano.config.floatX))
trainset['targets'] = theano.shared(np.asarray(trainset_raw['targets'], theano.config.floatX))

validset['hists_visual'] = theano.shared(np.asarray(validset_raw['hists_visual'], theano.config.floatX))
validset['hists_anno'] = theano.shared(validset_raw['hists_anno'].astype(theano.config.floatX))
validset['global_features'] = theano.shared(np.asarray(validset_raw['global_features'], theano.config.floatX))
validset['targets'] = theano.shared(np.asarray(validset_raw['targets'], theano.config.floatX))

testset['hists_visual'] = theano.shared(np.asarray(testset_raw['hists_visual'], theano.config.floatX))
testset['hists_anno'] = theano.shared(testset_raw['hists_anno'].astype(theano.config.floatX))
testset['global_features'] = theano.shared(np.asarray(testset_raw['global_features'], theano.config.floatX))
testset['targets'] = theano.shared(np.asarray(testset_raw['targets'], theano.config.floatX))

n_train = trainset_raw['meta']['length']
n_valid = validset_raw['meta']['length']
n_test = testset_raw['meta']['length']

n_train_batches = trainset_raw['meta']['length'] / batch_size
n_valid_batches = validset_raw['meta']['length'] / batch_size
n_test_batches = testset_raw['meta']['length'] / batch_size

aver_words_count_trainset = trainset_raw['hists_visual'].sum(axis=1).mean()
print 'average word counts of trainset is %f'%(aver_words_count_trainset)

model = SupDeepDocNADE(hidden_size = hidden_size,
                       learning_rate = learning_rate,
#                        learning_rate_unsup = learning_rate_unsup,
                       activation_function = activation_function,
                       word_representation_size = 0,
                       dropout_rate = dropout_rate,
                       normalize_by_document_size = normalize_by_document_size,
                       anno_weight = anno_weight,
                       batch_size = batch_size,
                       sup_option = sup_option,
                       unsup_weight = unsup_weight,
                       aver_words_count = aver_words_count_trainset,
                       preprocess_method = scaled_method,
                       length_limit = length_limit,
                       decrease_constant = decrease_constant,
                       polyakexp_weight = polyakexp_weight
                       )
pretrain_learning_rate = 0 # when it == 0, means no pretraining

spatial_split = np.asarray(spatial_pyramid, np.int32)**2*trainset_raw['meta']['voc_size']
region_split = np.append(spatial_split, trainset_raw['meta']['text_voc_size'])
region_split = np.add.accumulate(region_split)

model.initialize(trainset_raw['meta']['voc_size']*trainset_raw['meta']['n_regions'], 
                 trainset_raw['meta']['text_voc_size'], 
                 trainset_raw['meta']['global_feat_size'],
                 trainset_raw['meta']['n_classes'],
                 region_split)
    
if use_pretrain:
    full_path_premodel = os.path.expanduser(full_path_premodel)
    if os.path.isfile(full_path_premodel):
        model_file = open(full_path_premodel, 'rb')
        pre_model = cPickle.load(model_file)
        model_file.close()
        pre_model.add_activation()
    else:
        print 'ERROR: pretrained model not found'
        exit(-1)
    
    assert(pre_model.hidden_size == hidden_size)
    assert(pre_model.activation_function_name == activation_function)
    
    print '========================pre_trained model loaded successfully======================================='
    pretrain_learning_rate = pre_model.learning_rate
    model.add_supervised_layer(model.n_layers-1)
    model.add_top_layer(model.n_layers-1)
    model.copy_parameters(pre_model)


model.dec_learning_rate.set_value(model.learning_rate)    
model.compile_function(model.n_layers, trainset, validset)
# model.compile_compute_representation_function(model.n_layers, batch_size, trainset)
best_valid_error = -np.inf
best_valid_prec50 = -np.inf
best_epoch = 0
best_model = copy.deepcopy(model)
nb_of_epocs_without_improvement = 0
epoch = 0
print '\n### Training DeepDocNADE ###'
start_training_time = t.time()
while(epoch < max_iter and nb_of_epocs_without_improvement < look_ahead):
    epoch += 1
    print 'Epoch {0}'.format(epoch)
    print '\tTraining   ...',
    start_time = t.time()
    cost_train = []
    unsup_cost_train = []
    sup_cost_train = []
    prob_target_train = np.zeros((n_train, n_classes))
    for minibatch_index in range(n_train_batches):
#         cost_value,log_prob_target_value, unsup_cost_value, sup_cost_value, h_value, first_input_value, h_sup_value  = model.train(minibatch_index)
        cost_value,log_prob_target_value, unsup_cost_value, sup_cost_value  = model.train(minibatch_index)
        cost_train += [cost_value]
        unsup_cost_train += [unsup_cost_value]
        sup_cost_train += [sup_cost_value]
        prob_target_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = log_prob_target_value
    train_cost_error = np.asarray(cost_train).mean()
    train_unsup_cost_error = np.asarray(unsup_cost_train).mean()
    train_sup_cost_error = np.asarray(sup_cost_train).mean()
    train_map,train_prec50 = compute_MAP_Prec50(np.exp(prob_target_train), train_labels)
    print 'Train     :', get_done_text(start_time),  " MAP: {0:.6f}".format(train_map), " Prec@50: {0:.6f}".format(train_prec50), " Cost Error: {0:.6f}".format(train_cost_error) , " Unsup_Cost Error: {0:.6f}".format(train_unsup_cost_error), " Sup_Cost Error: {0:.6f}".format(train_sup_cost_error), 'mean_p: {0:.6f}'.format(np.exp(prob_target_train).mean()) 
    
    print '\tValidating ...',
    start_time = t.time()
    cost_valid = []
    unsup_cost_valid = []
    sup_cost_valid = []
    prob_target_valid = np.zeros((n_valid, n_classes))
    for minibatch_index in range(n_valid_batches):
#         cost_value,log_prob_target_value, unsup_cost_value, sup_cost_value, h_value, first_input_value , h_sup_value = model.valid(minibatch_index)
        cost_value,log_prob_target_value, unsup_cost_value, sup_cost_value = model.valid(minibatch_index)
        cost_valid += [cost_value]
        unsup_cost_valid += [unsup_cost_value]
        sup_cost_valid += [sup_cost_value]
        prob_target_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = log_prob_target_value
    valid_cost_error = np.asarray(cost_valid).mean()
    valid_unsup_cost_error = np.asarray(unsup_cost_valid).mean()
    valid_sup_cost_error = np.asarray(sup_cost_valid).mean()
    valid_map,valid_prec50 = compute_MAP_Prec50(np.exp(prob_target_valid), valid_labels)
    print 'Validation:', get_done_text(start_time),  " MAP: {0:.6f}".format(valid_map), " Prec@50: {0:.6f}".format(valid_prec50), " Cost Error: {0:.6f}".format(valid_cost_error) , " Unsup_Cost Error: {0:.6f}".format(valid_unsup_cost_error), " Sup_Cost Error: {0:.6f}".format(valid_sup_cost_error), 'mean_p: {0:.6f}'.format(np.exp(prob_target_valid).mean())  
    if valid_map > best_valid_error:
#         start_time = t.time()
        best_valid_error = valid_map
        best_valid_prec50 = valid_prec50
        best_epoch = epoch
        nb_of_epocs_without_improvement = 0
        del best_model
        gc.collect()
        best_model = copy.deepcopy(model)
#         print 'deep copying...',get_done_text(start_time)
    else:
        nb_of_epocs_without_improvement += 1

                    

print 'begin polyak svm part'    

#compute hidden representation of the testset
hidden_represenation_trainset = np.zeros((n_train, best_model.hidden_size[-1]))
hidden_represenation_validset = np.zeros((n_valid, best_model.hidden_size[-1]))
hidden_represenation_testset = np.zeros((n_test, best_model.hidden_size[-1]))
best_model.compile_compute_representation_function_polyak(best_model.n_layers, trainset)
for minibatch_index in range(n_train_batches):
    h,log_prob_target_value = best_model.compute_representation(minibatch_index)
    hidden_represenation_trainset[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = h
best_model.compile_compute_representation_function_polyak(best_model.n_layers, validset)
for minibatch_index in range(n_valid_batches):
    h,log_prob_target_value = best_model.compute_representation(minibatch_index)
    hidden_represenation_validset[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = h
best_model.compile_compute_representation_function_polyak(best_model.n_layers, testset)
prob_target_test = np.zeros((n_test, n_classes))
for minibatch_index in range(n_test_batches):
    h,log_prob_target_value = best_model.compute_representation(minibatch_index)
    hidden_represenation_testset[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = h
    prob_target_test[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :] = log_prob_target_value


hidden_represenation_trainset = hidden_represenation_trainset.tolist()
hidden_represenation_validset = hidden_represenation_validset.tolist()    
hidden_represenation_testset = hidden_represenation_testset.tolist()

#=========================================Logistic layer MAP===========================================

test_map_polyak,test_prec50_polyak = compute_MAP_Prec50(np.exp(prob_target_test), test_labels)
print 'Logistic MAP is '+ str(test_map_polyak)
print 'Logistic prec@50 is '+ str(test_prec50_polyak) 



#======================================================================================================

best_C_Linear_polyak = np.zeros(n_classes)
best_ap_Linear_polyak = -np.ones(n_classes)*np.inf
best_prec_Linear_polyak = -np.ones(n_classes)*np.inf
Linear_C = np.arange(Linear_minC, Linear_maxC, 0.25)


print 'Linear SVM Model Training'
cnt_Linear = 0
for C in Linear_C:
#     print cnt_Linear
    start = t.time()
    for i in range(n_classes):
        label_train = train_labels[:,i]
        label_train = label_train.tolist()
        label_valid = valid_labels[:,i]
        label_valid = label_valid.tolist()
        
        train_options = '-s 0 -c %e -q'%np.exp2(C)
        test_options = '-b 0 -q'
#         train_options = '-s 1 -c %e -q'%np.exp2(C)
#         test_options = '-q'
        svm_model = train(label_train, hidden_represenation_trainset, train_options)
        
        p_labels, p_acc, p_vals = predict(label_valid, hidden_represenation_validset, svm_model, test_options)
        p_vals = np.asarray(p_vals)
        index = svm_model.get_labels().index(1)
        if index ==0:
            confidence = p_vals
        elif index == 1:
            confidence = -p_vals
        else:
            raise TypeError('wrong index')
#         confidence = p_vals[:, index]
        this_ap, this_prec = compute_AP_Prec50(confidence, valid_labels[:,i])
        if this_ap > best_ap_Linear_polyak[i]:
            best_ap_Linear_polyak[i] = this_ap
            best_prec_Linear_polyak[i] = this_prec
            best_C_Linear_polyak[i] = C 
    end = t.time()
    print '%d/%d cross-validation cost time %f'%(cnt_Linear, len(Linear_C), end-start)
    print 'the map for now on validset is %f'%(np.mean(best_ap_Linear_polyak))
    cnt_Linear += 1


print '=======================================Final SVM Part==============================================='
Linear_ap_list = []
Linear_prec_list = []
Linear_ap = 0
Linear_prec = 0
hidden_represenation_trainset.extend(hidden_represenation_validset)
train_labels_final = np.vstack((train_labels, valid_labels))
# file_conf = open("/home/local/USHERBROOKE/zhey2402/DeepDocNADE/SupDocNADE_Confidence_value.txt", 'w')
for i in range(n_classes):
    print 'Final SVM for class %d'%i
    label_train = train_labels_final[:,i]
    label_train = label_train.tolist()
    label_test = test_labels[:,i]
    label_test = label_test.tolist()
    
#     train_options = '-s 1 -c %e -q'%np.exp2(best_C_Linear[i])
#     test_options = '-q'
    train_options = '-s 0 -c %e -q'%np.exp2(best_C_Linear_polyak[i])
    test_options = '-b 0 -q'
    svm_model = train(label_train, hidden_represenation_trainset, train_options)
    p_labels, p_acc, p_vals = predict(label_test, hidden_represenation_testset, svm_model, test_options)
    p_vals = np.asarray(p_vals)
    index = svm_model.get_labels().index(1)
    if index ==0:
        confidence = p_vals
    elif index == 1:
        confidence = -p_vals
    else:
        raise TypeError('wrong index')
#     confidence = p_vals[:, index]
    this_ap, this_prec = compute_AP_Prec50(confidence, test_labels[:,i])
    Linear_ap += this_ap
    Linear_prec += this_prec
    Linear_ap_list.append(this_ap)
    Linear_prec_list.append(this_prec)
#     confidence.tofile(file_conf, sep=' ', format='%s')
#     file_conf.write('\n')
# file_conf.close()
Linear_map_polyak = Linear_ap/n_classes
Linear_prec50_polyak = Linear_prec/n_classes        
print 'Linear SVM map is '+ str(Linear_map_polyak)
print 'Linear SVM prec@50 is '+ str(Linear_prec50_polyak) 
#===============================================================================      


line_linear = '%f %f %f %f %f %f %f %f %d %s %s %s %s %d %d %d %f %f %s %s %f %s %s %f %f %f %s %s\n'%(Linear_map_polyak,
                                                                                                       np.mean(best_ap_Linear_polyak),
                                                                                                       test_map_polyak,
                                                                                                       best_valid_error,
                                                                                                       Linear_prec50_polyak,
                                                                                                       np.mean(best_prec_Linear_polyak),
                                                                                                       test_prec50_polyak, 
                                                                                                       best_valid_prec50,
                                                                                                       folder_ID, 
                                                                                                       spatial_pyramid,
                                                                                                       hidden_size,
                                                                                                       learning_rate,
                                                                                                       activation_function,
                                                                                                       max_iter, 
                                                                                                       look_ahead,
                                                                                                       epoch,
                                                                                                       Linear_minC, 
                                                                                                       Linear_maxC,
                                                                                                       dropout_rate,
                                                                                                       unsup_weight,
                                                                                                       anno_weight,
                                                                                                       sup_option,
                                                                                                       scaled_method,
                                                                                                       length_limit, 
                                                                                                       decrease_constant,
                                                                                                       polyakexp_weight,
                                                                                                       ' '.join(str(x) for x in best_C_Linear_polyak),
                                                                                                       full_path_premodel
                                                                                                       )
uniresultfile_linear = open(uniresultfile_name_Linear, 'a')
fcntl.flock(uniresultfile_linear.fileno(), fcntl.LOCK_EX)
uniresultfile_linear.write(line_linear)
uniresultfile_linear.close() # unlocks the file




print 'done'
