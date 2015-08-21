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
    unlabel_size = int(sizes_file.readline())
    sizes_file.close()
#     lengths = [unlabel_size]
    meta_file = open(os.path.join(dir_path, 'meta.txt'))
    meta = {}
    for line in meta_file:
        meta_name, meta_value = line.rstrip().split(':')
        meta.update({meta_name:int(meta_value)})
        
    spatial_split = np.asarray(spatial_split, np.float64)
    n_regions = int((spatial_split**2).sum())
    meta['n_regions'] = n_regions  
    
    unlabel_size = int(unlabel_size/meta['dataset_split'])    
    file_annotations = os.path.join(dir_path, 'text', 'text_all_2000_unlabelled.npz')
    annotations = LoadSparse(file_annotations, verbose = False)
    
    unlabel_str,length_str = 'unlabeled','length'
    unlabel_file = os.path.join(dir_path,unlabel_str+str(folder_ID)+'.npz')
    unlabel_meta = {length_str:unlabel_size}
    unlabel_meta.update(meta)
    unlabel_meta.update({'Folder_ID':folder_ID})

    npzfile_train = np.load(unlabel_file)
    if log_option == 'NoLog':
        unlabel_hists = npzfile_train['unlabeled_matrix_hists'][:,:n_regions*unlabel_meta['voc_size']]
    elif log_option == 'Log_Natural':
        unlabel_hists = np.round(np.log(npzfile_train['unlabeled_matrix_hists'][:,:n_regions*unlabel_meta['voc_size']]+1.0)).astype(np.int32)
    elif log_option == 'Log_4':
        unlabel_hists = np.round(np.log(npzfile_train['unlabeled_matrix_hists'][:,:n_regions*unlabel_meta['voc_size']]+1.0)/np.log(4)+np.finfo(np.double).eps).astype(np.int32)
#     unlabel_hists = npzfile_train['unlabeled_matrix_hists']
    unlabel_global_features = npzfile_train['unlabeled_matrix_global_features']
    unlabel_annos = annotations[(folder_ID-1)*unlabel_size:folder_ID*unlabel_size,:]
    

    return {'hists_visual':unlabel_hists, 'global_features':unlabel_global_features, 'hists_anno':unlabel_annos,'meta':unlabel_meta}

def obtain(dir_path):
    """
    Gives information about how to obtain this dataset (``dir_path`` is ignored).
    """

    print 'Ask Yin Zheng (yzheng3xg@gmail.com) for the data.'

