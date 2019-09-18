
import os
import math
import numpy as np
import csv
import ast
from skimage.measure import block_reduce
from scipy.ndimage.interpolation import rotate, shift
import nibabel as nib

class Data(object):
    '''
    Class used to load data for ISLES, BRATS and IXI datasets. ISLES2015 and BRATS data can be downloaded from
    http://www.isles-challenge.org/ISLES2015 and https://sites.google.com/site/braintumorsegmentation/home/brats2015.
    This loader expects the data to be in a .npz format where there's one .npz file for each modality, e.g. T1.npz, T2.npz etc.
    Each .npz contains an array of volumes, and every volume has shape (Z, H, W), where Z is the number of spatial slices 
    and H, W the height and width of a slice respectively.
    IXI dataset can be downloaded from http://brain-development.org/ixi-dataset and can be loaded as-is.

    A splits.txt file is expected if the Data object will be used for cross-validation through runner.py.
    An example splits.txt could have the following contents:
    test,validation,train
    "[0,1]","[2,3]","[4,5]"

    Example usage of the class:
    data = Data('./data/IXI', modalities_to_load=['T1','T2'], dataset='ISLES', trim_and_downsample=False)
    data.load()
    vols_0_1_2 = data.select_for_ids('T1', [0, 1, 2])
    '''
    def __init__(self, data_folder, modalities_to_load=None):
        self.data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder

        self.num_vols = 14
        self.splits_file = './splits.txt'

        if modalities_to_load is not None:
            self.modalities_to_load = modalities_to_load
        else:
            self.modalities_to_load = ['T2', 'PD']

        self.T1 = None
        self.T2 = None
        self.VFlair = None
        self.MASK = None

        self.channels = dict()
        # self.refDict = {'T1': self.T1, 'T2': self.T2, 'MASK': self.MASK}
        self.refDict = {'T1': self.T1, 'T2': self.T2, 'VFlair': self.VFlair, 'MASK': self.MASK}

    def load(self):
        for mod_name in self.modalities_to_load:
            print 'Loading ' + mod_name
            norm_vols = False if mod_name == 'MASK' else True
            mod = self.load_modality(mod_name, normalize_volumes=norm_vols)

            self.refDict[mod_name] = mod

        # self.T1 = self.refDict['T1']
        # self.T2 = self.refDict['T2']
        # self.VFlair = self.refDict['VFlair']
        # self.MASK = self.refDict['MASK']


    # def remove_volume(self, vol):
    #     if self.T1 is not None:
    #         del self.T1[vol]
    #     if self.T2 is not None:
    #         del self.T2[vol]
    #     if self.VFlair is not None:
    #         del self.VFlair[vol]
    #     if self.MASK is not None:
    #         del self.MASK[vol]

    #     self.num_vols -= 1

    def load_ixi(self, mod):
        folder = self.data_folder + '/IXI-' + mod
        data = [nib.load(folder + '/' + f).get_data() for f in np.sort(os.listdir(folder))]
        # data = [np.swapaxes(np.swapaxes(d, 1, 2), 0, 1) for d in data]
        # print data[0].shape
        # image_path = folder + '/'
        # image_list = ['bet-ms0056-05.nii.gz','bet-ms0060-01.nii.gz'] 
        image = []
        for d in data:
            image_array = []
            # data = nib.load(os.path.join(image_path, image_name)).get_data()
            for i in range(50,180):
                dummy_image = d[:,i,:]          
                image_array.append(dummy_image)   # Here we create a 3d matrix from several slices
            image = image + [np.array(image_array)]       # Here we create a 4d matrix from multiple input images
        if mod == 'LPM':
            image_array = image[0]
            for i in range(1,self.num_vols):
                image = image + [image_array]
        print 'Loaded %d vols from IXI' % len(data)
        return image

    def load_modality(self, modality, normalize_volumes=True):

        data = self.load_ixi(modality)

        # array of 3D volumes 
        # To Do: normalize based on slices not volume
        X = [data[i].astype('float32') for i in range(self.num_vols)]

        if normalize_volumes:
            for i, x in enumerate(X):
                X[i] = X[i] / x[x!=0].mean()

        return X

    def select_for_ids(self, modality, ids, as_array=True):
        # assert modality in self.refDict
        print ids
        print modality
        print "Length : %d"  % len (self.refDict[modality])
        data_ids = [self.refDict[modality][i] for i in ids]

        if as_array:
            data_ids_ar = np.concatenate(data_ids)
            if len(data_ids_ar.shape) < 4:
                data_ids_ar = np.expand_dims(data_ids_ar, axis=1)
            if modality in self.channels:
                ch_ids = [self.refDict[self.channels[modality]][i] for i in ids]
                ch_ids_ar = np.expand_dims(np.concatenate(ch_ids), axis=1)
                return np.concatenate([data_ids_ar, ch_ids_ar], axis=1)
            else:
                return data_ids_ar
        else:
            data_ids_ar = data_ids
            if len(data_ids_ar[0].shape) < 4:
                data_ids_ar = [np.expand_dims(d, axis=1) for d in data_ids]
            if modality in self.channels:
                ch_ids = [self.refDict[self.channels[modality]][i] for i in ids]
                ch_ids_ar = [np.expand_dims(ch, axis=1) for ch in ch_ids]
                return [np.concatenate([data_ids_ar[i], ch_ids_ar[i]], axis=1) for i in range(len(ids))]
            else:
                return data_ids_ar

    def id_splits_iterator(self):
        # return a dictionary of train, validation and test ids
        with open(self.splits_file, 'r') as f:
            r = csv.reader(f, delimiter=',', quotechar='"')
            headers = next(r)
            for row in r:
                if len(row) == 0:
                    break
                if row[0].startswith('#'):
                    continue

                yield {headers[i].strip(): ast.literal_eval(row[i]) for i in range(len(headers))}
