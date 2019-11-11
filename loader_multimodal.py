
import os
import math
import numpy as np
import csv
import ast
from skimage.measure import block_reduce
from scipy.ndimage.interpolation import rotate, shift
import nibabel as nib
import glob
import uuid

from PIL import Image
import cv2
from scipy import misc
import numpy as np

from skimage.segmentation import slic
from skimage.measure import regionprops


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
    def __init__(self, data_folder, modalities_to_load=None, trim_and_downsample=False):
        self.data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder

        self.num_vols =270
        self.splits_file = './splits.txt'

        if modalities_to_load is not None:
            self.modalities_to_load = modalities_to_load
        else:
            self.modalities_to_load = ['T2', 'PD']

        self.T1 = None
        self.T2 = None
        self.VFlair = None
        self.MASK = None
        # self.LPM= None

        self.channels = dict()
        self.rotations = {mod: False for mod in self.modalities_to_load}
        self.shifts = {mod: False for mod in self.modalities_to_load}
        self.refDict = {'T1': self.T1, 'T2': self.T2, 'MASK': self.MASK}
        #for synthesis
        # self.refDict = {'T1': self.T1, 'T2': self.T2}
        self.trim_and_downsample = trim_and_downsample
        # self.refDict = {'T1': self.T1, 'T2': self.T2, 'VFlair': self.VFlair, 'MASK': self.MASK}

    def load(self):
        for mod_name in self.modalities_to_load:
            # print 'Loading ' + mod_name
            norm_vols = False if mod_name == 'MASK' else True
            mod = self.load_modality(mod_name, normalize_volumes=norm_vols,rotate_mult=self.rotations[mod_name],
                                     shift_mult=self.shifts[mod_name])

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
        data = [np.swapaxes(np.swapaxes(d, 1, 2), 0, 1) for d in data]
        print data[0].shape
        # return data
        # image_path = folder + '/'
        # image_list = ['bet-ms0056-05.nii.gz','bet-ms0060-01.nii.gz'] 
        image = []
        for img in data:
            
            # im = misc.imread(img)
            # im = im.convert('L')
            # final_image=[]
            # create_size= img.shape
            # size =  (create_size[1],create_size[2])

            
            # for i in range(img.shape[0]):
                
                #im=Image.frombytes('L',size,img[i])
                #im = im.resize((im.size[0]//8, im.size[1]//8))
                # img= list(im)

                # gt = Image.open(os.path.splitext(img)[0] + '.gz')
                # gt = gt.resize(im.size)
                # # we don't care about border pixels
                # gt = np.array(gt)[:,:,-1]

            #     sp = slic(img[i], n_segments=8)
            #     # extract all centroids
            #     props = regionprops(sp)

            #     # extract 28x28 image patch from input and determine class for this
            #     # superpixel
            
            #     for prop in props:
                
            #         y = int(prop.centroid[0])
            #         x = int(prop.centroid[1])
            #         siz = 14
            #         patch = im.crop((x-siz, y-siz, x+siz, y+siz))
            #         temp_slice.append(patch)
            
            # final_image.append(temp_slice)
            image_array = []
            # data = nib.load(os.path.join(image_path, image_name)).get_data()
        # image_array=[]
            for i in range(125,135):
                dummy_image = img[:,:,i]          
                image_array.append(dummy_image)   # Here we create a 3d matrix from several slices
            image = image + [np.array(image_array)]       # Here we create a 4d matrix from multiple input images
        # if mod == 'LPM':
        #     patch = image[0]
        #     for i in range(1,self.num_vols):
        #         image = image + [patch]
        # print 'Loaded %d vols from IXI' % len(data)
        
        # return final_image,mod
        return image


    def load_modality(self, modality, normalize_volumes=True, downsample=2, rotate_mult=0.0, shift_mult=0.0):

        # enc= [self.load_ixi(m) for m in self.modalities_to_load]
        # data = [final_image for (final_image, mod) in enc]
        # data = [self.load_ixi(m) for m in self.modalities_to_load]
        data=self.load_ixi(modality)

        # array of 3D volumes 
        # To Do: normalize based on slices not volume
        #converting to float
        # for i in data:
        #     float(i)
        X = [data[i] for i in range(self.num_vols)]

        # trim the matrices and downsample: downsample x downsample -> 1x1
        for i, x in enumerate(X):
            if rotate_mult != 0:
                print 'Rotating ' + modality + '. Multiplying by ' + str(rotate_mult)
                rotations = [[-5.57, 2.79, -11.99], [-5.42, -18.34, -14.22], [4.64, 5.80, -5.96],
                             [-17.02, -8.70, 15.43],
                             [18.79, 17.44, 17.06], [-14.55, -4.90, 9.19], [14.37, -0.58, -16.85],
                             [-9.49, -12.53, -2.89],
                             [-16.75, -4.07, 3.23], [14.39, -16.58, 3.35], [-14.05, -2.25, -10.58],
                             [8.47, -8.95, -12.73],
                             [13.00, -10.90, -2.85], [2.61, -7.51, -6.26], [-13.99, -0.38, 6.29],
                             [10.16, -9.88, -11.89],
                             [6.76, 0.83, -19.85], [18.74, -6.70, 15.46], [-3.01, -2.85, 18.45], [-17.37, -1.32, -3.48],
                             [14.67, -17.93, 18.74], [6.55, 18.19, -8.24], [13.52, -4.09, 19.32], [5.27, 11.27, 4.93],
                             [2.29, 17.83, 10.07], [-11.98, 10.49, 0.02], [14.49, -12.00, -17.21],
                             [17.86, -17.38, 19.04]]
                theta = rotations[i]

                x = rotate(x, rotate_mult * theta[0], axes=(1, 0), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)
                x = rotate(x, rotate_mult * theta[1], axes=(1, 2), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)
                x = rotate(x, rotate_mult * theta[2], axes=(0, 2), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)

            if shift_mult != 0:
                print 'Shifting ' + modality + '. Multiplying by ' + str(shift_mult)
                shfts = [[0.931, 0.719, -0.078], [0.182, -0.220, 0.814], [0.709, 0.085, -0.262], [-0.898, 0.367, 0.395],
                         [-0.936, 0.591, -0.101], [0.750, 0.522, 0.132], [-0.093, 0.188, 0.898],
                         [-0.517, 0.905, -0.389],
                         [0.616, 0.599, 0.098], [-0.209, -0.215, 0.285], [0.653, -0.398, -0.153],
                         [0.428, -0.682, -0.501],
                         [-0.421, -0.929, -0.925], [-0.753, -0.492, 0.744], [0.532, -0.302, 0.353],
                         [0.139, 0.991, -0.086],
                         [-0.453, 0.657, 0.072], [0.576, 0.918, 0.242], [0.889, -0.543, 0.738], [-0.307, -0.945, 0.093],
                         [0.698, -0.443, 0.037], [-0.209, 0.882, 0.014], [0.487, -0.588, 0.312],
                         [0.007, -0.789, -0.107],
                         [0.215, 0.104, 0.482], [-0.374, 0.560, -0.187], [-0.227, 0.030, -0.921], [0.106, 0.975, 0.997]]
                shft = shfts[i]
                x = shift(x, [shft[0] * shift_mult, shft[1] * shift_mult, shft[2] * shift_mult])

            if self.trim_and_downsample:
                X[i] = block_reduce(x, block_size=(1, downsample, downsample), func=np.mean)

                if self.dataset == 'BRATS':
                    # power of 2 padding
                    (_, w, h) = X[i].shape

                    w_pad_size = int(math.ceil((math.pow(2, math.ceil(math.log(w, 2))) - w) / 2))
                    h_pad_size = int(math.ceil((math.pow(2, math.ceil(math.log(h, 2))) - h) / 2))

                    X[i] = np.lib.pad(X[i], ((0, 0), (w_pad_size, w_pad_size), (h_pad_size, h_pad_size)), 'constant',
                                      constant_values=0)

                    (_, w, h) = X[i].shape

                    # check if dimensions are even

                    if w & 1:
                        X[i] = X[i][:, 1:, :]

                    if h & 1:
                        X[i] = X[i][:, :, 1:]


            else:
                X[i] = x

        if normalize_volumes:
            for i, x in enumerate(X):
                X[i] = X[i] / np.mean(x)

        if rotate_mult > 0:
            for i, x in enumerate(X):
                X[i][X[i] < 0.25] =0

        return X

    def add_channel(self, modality, channel):
        assert modality in self.refDict
        assert channel in self.refDict
        self.channels.update({modality: channel})

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

    

            
