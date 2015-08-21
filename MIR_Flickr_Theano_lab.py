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

"""
Module ``datasets.MIR_Flickr`` gives access to the MIR_Flickr dataset (labeled part) for Theano.



"""
import numpy as np
import os
import scipy.sparse as sp

def LoadSparse(inputfile, verbose=False):
    """Loads a sparse matrix stored as npz file."""
    npzfile = np.load(inputfile)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                         npzfile['indptr']),
                        shape=tuple(list(npzfile['shape'])))
    if verbose:
        
        print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                          mat.shape.__str__())
    return mat

def load(dir_path,folder_ID, log_option='NoLog', spatial_split=[1,2,3]):
    """
    """

    dir_path = os.path.expanduser(dir_path)
    sizes_file = open(os.path.join(dir_path,'sizes.txt'),'r')
    train_size,valid_size,test_size = int(sizes_file.readline()),int(sizes_file.readline()),int(sizes_file.readline())
    sizes_file.close()
    lengths = [train_size,valid_size,test_size]
    meta_file = open(os.path.join(dir_path, 'meta.txt'))
    meta = {}
    for line in meta_file:
        meta_name, meta_value = line.rstrip().split(':')
        meta.update({meta_name:int(meta_value)})
        
    spatial_split = np.asarray(spatial_split, np.float64)
    n_regions = int((spatial_split**2).sum())
    meta['n_regions'] = n_regions  
    file_train_indices = os.path.join(dir_path, 'splits', 'train_indices_'+str(folder_ID)+'.npy' )
    file_valid_indices = os.path.join(dir_path, 'splits', 'valid_indices_'+str(folder_ID)+ '.npy' )
    file_test_indices = os.path.join(dir_path, 'splits', 'test_indices_'+str(folder_ID)+'.npy' )
    train_indices = np.load(file_train_indices)
    valid_indices = np.load(file_valid_indices)
    test_indices = np.load(file_test_indices)
    file_annotations = os.path.join(dir_path, 'text', 'text_all_2000_labelled.npz')
    annotations = LoadSparse(file_annotations, verbose = True)
    
    train_str,valid_str,test_str,length_str = 'train','valid','test','length'
    train_file,valid_file,test_file = [os.path.join(dir_path,name+str(folder_ID)+'.npz') for name in [train_str,valid_str,test_str]]
    train_meta,valid_meta,test_meta = [{length_str:length} for length in lengths]
    train_meta.update(meta)
    valid_meta.update(meta)
    test_meta.update(meta)

    npzfile_train = np.load(train_file)
    if log_option == 'NoLog':
        trainset_hists = npzfile_train['trainset_matrix_hists'][:,:n_regions*train_meta['voc_size']]
    elif log_option == 'Log_Natural':
        trainset_hists = np.round(np.log(npzfile_train['trainset_matrix_hists'][:,:n_regions*train_meta['voc_size']]+1.0)).astype(np.int32)
    elif log_option == 'Log_4':
        trainset_hists = np.round(np.log(npzfile_train['trainset_matrix_hists'][:,:n_regions*train_meta['voc_size']]+1.0)/np.log(4)+np.finfo(np.double).eps).astype(np.int32)
        
    trainset_global_features = npzfile_train['trainset_matrix_global_features']
    trainset_targets = npzfile_train['trainset_matrix_targets']
    trainset_annos = annotations[train_indices, :]
    
    npzfile_valid = np.load(valid_file)
    if log_option == 'NoLog':
        validset_hists = npzfile_valid['validset_matrix_hists'][:,:n_regions*valid_meta['voc_size']]
    elif log_option == 'Log_Natural':
        validset_hists = np.round(np.log(npzfile_valid['validset_matrix_hists'][:,:n_regions*valid_meta['voc_size']]+1.0)).astype(np.int32)
    elif log_option == 'Log_4':
        validset_hists = np.round(np.log(npzfile_valid['validset_matrix_hists'][:,:n_regions*valid_meta['voc_size']]+1.0)/np.log(4)+np.finfo(np.double).eps).astype(np.int32)
#     validset_hists = npzfile_valid['validset_matrix_hists'][:,:n_regions]
    validset_global_features = npzfile_valid['validset_matrix_global_features']
    validset_targets = npzfile_valid['validset_matrix_targets']
    validset_annos = annotations[valid_indices, :]
    
    npzfile_test = np.load(test_file)
    if log_option == 'NoLog':
        testset_hists = npzfile_test['testset_matrix_hists'][:,:n_regions*test_meta['voc_size']]
    elif log_option == 'Log_Natural':
        testset_hists = np.round(np.log(npzfile_test['testset_matrix_hists'][:,:n_regions*test_meta['voc_size']]+1.0)).astype(np.int32)
    elif log_option == 'Log_4':
        testset_hists = np.round(np.log(npzfile_test['testset_matrix_hists'][:,:n_regions*test_meta['voc_size']]+1.0)/np.log(4)+np.finfo(np.double).eps).astype(np.int32)
#     testset_hists = npzfile_test['testset_matrix_hists'][:,:n_regions]
    testset_global_features = npzfile_test['testset_matrix_global_features']
    testset_targets = npzfile_test['testset_matrix_targets']
    testset_annos = annotations[test_indices, :]

    return ({train_str:{'hists_visual':trainset_hists, 'global_features':trainset_global_features, 'targets':trainset_targets, 'hists_anno':trainset_annos,'meta':train_meta},
             valid_str:{'hists_visual':validset_hists, 'global_features':validset_global_features, 'targets':validset_targets, 'hists_anno':validset_annos,'meta':valid_meta},
             test_str:{'hists_visual':testset_hists, 'global_features':testset_global_features, 'targets':testset_targets, 'hists_anno':testset_annos,'meta':test_meta}})

def obtain(dir_path):
    """
    Gives information about how to obtain this dataset (``dir_path`` is ignored).
    """

    print 'Ask Yin Zheng (yzheng3xg@gmail.com) for the data.'

