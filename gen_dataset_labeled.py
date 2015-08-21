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

@author: yin.zheng
'''

import numpy as np
import os
import string
import time
import scipy.sparse as sp
import collections
from itertools import izip 

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

def load(dir_path, path_output_data, folder_ID):
        """
        dir_path: The dataset to the extracted folder from data download from http://www.cs.toronto.edu/~nitish/multimodal/index.html
        path_output_data: The path to save the processed dataset
        folder_ID: the split ID (from 1 to 5)

        """

        dir_path = os.path.expanduser(dir_path)
        file_train_indices = os.path.join(dir_path, 'splits', 'train_indices_'+str(folder_ID)+'.npy' )
        file_valid_indices = os.path.join(dir_path, 'splits', 'valid_indices_'+str(folder_ID)+ '.npy' )
        file_test_indices = os.path.join(dir_path, 'splits', 'test_indices_'+str(folder_ID)+'.npy' )
        train_indices = np.load(file_train_indices)
        valid_indices = np.load(file_valid_indices)
        test_indices = np.load(file_test_indices)
        
        file_labels = os.path.join(dir_path, 'labels.npy')
        labels = np.load(file_labels).astype(np.int32)
        trainset_matrix_targets = labels[train_indices, :]
        validset_matrix_targets = labels[valid_indices, :]
        testset_matrix_targets = labels[test_indices, :]
        
        file_annotations = os.path.join(dir_path, 'text', 'text_all_2000_labelled.npz')
        annotations = LoadSparse(file_annotations, verbose = True)
        train_annos = annotations[train_indices, :]
        valid_annos = annotations[valid_indices, :]
        test_annos = annotations[test_indices, :]
        
        
        file_global_features_unlab = os.path.join(dir_path, 'image', 'unlabelled', 'combined-00003_1-of-00100.npy')
        global_features_unlab = np.load(file_global_features_unlab)[:, :-2000]
        for i in range(2):
            if i+4<10:
                tmp_file_global_features_unlab = os.path.join(dir_path, 'image', 'unlabelled', 'combined-0000'+str(i+4)+'-of-00100.npy')
            elif i+4 <100:
                tmp_file_global_features_unlab = os.path.join(dir_path, 'image', 'unlabelled', 'combined-000'+str(i+4)+'-of-00100.npy')
            else:
                tmp_file_global_features_unlab = os.path.join(dir_path, 'image', 'unlabelled', 'combined-00'+str(i+4)+'-of-00100.npy')
            tmp_global_features_unlab = np.load(tmp_file_global_features_unlab)[:, :-2000]
            global_features_unlab = np.vstack((global_features_unlab, tmp_global_features_unlab))
            
        mean_global_features_unlab = np.mean(a=global_features_unlab, axis=0, dtype=np.float64)
        std_global_features_unlab = np.std(a=global_features_unlab, axis=0, dtype=np.float64)    
        del global_features_unlab
        
        file_global_features1 = os.path.join(dir_path, 'image', 'labelled', 'combined-00001-of-00100.npy')
        global_features1 = np.load(file_global_features1)
        file_global_features2 = os.path.join(dir_path, 'image', 'labelled', 'combined-00002-of-00100.npy')
        global_features2 = np.load(file_global_features2)
        file_global_features3 = os.path.join(dir_path, 'image', 'labelled', 'combined-00003_0-of-00100.npy')
        global_features3 = np.load(file_global_features3)
        global_features = np.vstack((global_features1, global_features2, global_features3))
        train_global_features = global_features[train_indices, :-2000]
        valid_global_features = global_features[valid_indices, :-2000]
        test_global_features = global_features[test_indices, :-2000] 
        
        trainset_matrix_hists = global_features[train_indices, -2000:]
        validset_matrix_hists = global_features[valid_indices, -2000:]
        testset_matrix_hists = global_features[test_indices, -2000:]
        
        
        train_global_features -= mean_global_features_unlab[np.newaxis,:]
        trainset_matrix_global_features = train_global_features / std_global_features_unlab[:, np.newaxis]
        
        valid_global_features -= mean_global_features_unlab[np.newaxis,:]
        validset_matrix_global_features = valid_global_features / std_global_features_unlab[:, np.newaxis]
        
        test_global_features -= mean_global_features_unlab[np.newaxis,:]
        testset_matrix_global_features = test_global_features / std_global_features_unlab[:, np.newaxis]
        
        file_train = os.path.join(path_output_data, 'train'+str(folder_ID))
        file_valid = os.path.join(path_output_data, 'valid'+str(folder_ID))
        file_test = os.path.join(path_output_data, 'test'+str(folder_ID))
        
        np.savez(file_train, trainset_matrix_hists=trainset_matrix_hists, trainset_matrix_global_features = trainset_matrix_global_features, trainset_matrix_targets=trainset_matrix_targets)
        np.savez(file_valid, validset_matrix_hists=validset_matrix_hists, validset_matrix_global_features = validset_matrix_global_features, validset_matrix_targets=validset_matrix_targets)
        np.savez(file_test, testset_matrix_hists=testset_matrix_hists, testset_matrix_global_features = testset_matrix_global_features, testset_matrix_targets=testset_matrix_targets)
        
        
        
 
if __name__ == "__main__":
    load('/media/2TDisk/Flickr', '/media/2TDisk/Flickr', 1)
    load('/media/2TDisk/Flickr', '/media/2TDisk/Flickr', 2)
    load('/media/2TDisk/Flickr', '/media/2TDisk/Flickr', 3)
    load('/media/2TDisk/Flickr', '/media/2TDisk/Flickr', 4)
    load('/media/2TDisk/Flickr', '/media/2TDisk/Flickr', 5)
