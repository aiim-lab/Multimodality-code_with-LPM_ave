# -*- coding: utf-8 -*- 
from loader_multimodal import Data
from model import Multimodel
from runner import Experiment

data = Data('./data/', modalities_to_load=['T1','T2','MASK','FLAIR'])
#for synthesis
# data = Data('./data/', modalities_to_load=['T1','T2','FLAIR'])
data.load()
input_modalities= ['T1','T2','MASK']
output_weights= {'FLAIR':1.0}
# output_weights= {'FLAIR':1.0}
exp = Experiment(input_modalities, output_weights, './', data, latent_dim=4, spatial_transformer= False, common_merge='ave', ind_outs=True, fuse_outs=True)
exp.run(data)     

# m = Multimodel(['T1','T2'], ['T2'],{ 'T2':1.0, 'concat':1.0} ,16, 256, 'max', True, True)
# m.build()
