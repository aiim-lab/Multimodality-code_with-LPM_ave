# -*- coding: utf-8 -*- 
from loader_multimodal import Data
from model import Multimodel
from runner import Experiment

data = Data('./data/', modalities_to_load=['T1','T2','FLAIR','MASK'])
data.load()
input_modalities= ['T1','T2']
output_weights= {'FLAIR':1.0,'MASK':1.0}
exp = Experiment(input_modalities, output_weights, './', data, latent_dim=4, common_merge='max', ind_outs=True, fuse_outs=True)
exp.run(data)

# m = Multimodel(['T1','T2'], ['T2'],{ 'T2':1.0, 'concat':1.0} ,16, 256, 'max', True, True)
# m.build()
