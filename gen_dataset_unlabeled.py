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

def load(dir_path, path_output_data):
    """
    ir_path: The dataset to the extracted folder from data download from http://www.cs.toronto.edu/~nitish/multimodal/index.html
    path_output_data: The path to save the processed dataset

    """

    dir_path = os.path.expanduser(dir_path)
    file_annotations = os.path.join(dir_path, 'text', 'text_all_2000_unlabelled.npz')
    annotations = LoadSparse(file_annotations, verbose = True)
    
    
    
    
    file_global_features = os.path.join(dir_path, 'image', 'unlabelled', 'combined-00003_1-of-00100.npy')
    global_features = np.load(file_global_features)[:, :-2000]
    unlabeled_matrix_hists = np.load(file_global_features)[:, -2000:]
    for i in range(97):
        if i+4<10:
            tmp_file_global_features = os.path.join(dir_path, 'image', 'unlabelled', 'combined-0000'+str(i+4)+'-of-00100.npy')
        elif i+4 <100:
            tmp_file_global_features = os.path.join(dir_path, 'image', 'unlabelled', 'combined-000'+str(i+4)+'-of-00100.npy')
        else:
            tmp_file_global_features = os.path.join(dir_path, 'image', 'unlabelled', 'combined-00'+str(i+4)+'-of-00100.npy')
        tmp_global_features = np.load(tmp_file_global_features)[:, :-2000]
        tmp_mir_unlab_histograms = np.load(tmp_file_global_features)[:, -2000:]
        global_features = np.vstack((global_features, tmp_global_features))
        unlabeled_matrix_hists = np.vstack((unlabeled_matrix_hists, tmp_mir_unlab_histograms))
        
    mean_global_features = np.mean(a=global_features, axis=0, dtype=np.float64)
    std_global_features = np.std(a=global_features, axis=0, dtype=np.float64)    
    global_features -= mean_global_features[np.newaxis,:]
    global_features /= std_global_features
    batch_size = global_features.shape[0]/50
    for i in range(50):
        file_unlab = os.path.join(path_output_data, 'unlabeled'+str(i+1))
        np.savez(file_unlab, unlabeled_matrix_hists=unlabeled_matrix_hists[batch_size*i:batch_size*(i+1),:], unlabeled_matrix_global_features = unlabeled_matrix_global_features[batch_size*i:batch_size*(i+1),:])
#         
        
        
    
    
                              
    
    
if __name__ == "__main__":
    load('/run/media/ian/2TDisk/Flickr', '/run/media/ian/2TDisk/Flickr')
