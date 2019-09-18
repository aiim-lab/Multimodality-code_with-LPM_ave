# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib

matplotlib.use('Agg')
import sys

sys.setrecursionlimit(10000)

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, merge, Lambda, LeakyReLU, MaxPooling2D,concatenate, Concatenate, maximum,average,add,Reshape, Multiply, Add

from keras.layers.convolutional import Conv2D, UpSampling2D, Conv3D
from keras import backend as K
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import sigmoid
import tensorflow as tf
from keras.layers import Dropout

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function



class Multimodel(object):
    '''
    Class for constructing a neural network model as described in
    T. Joyce, A. Chartsias, S.A. Tsaftaris, 'Robust Multi-Modal MR Image Synthesis,' MICCAI 2017

    The compiled Keras model inputs and outputs are the following:
    inputs: list of numpy data arrays, one for each modality
    outputs: list containing numpy arrays for each output modality, 2 zero numpy arrays (one used for variance minimisation,
    the other as a dummy value since the last output of the model contains latent representations)

    The model is 2D, so the input numpy arrays are of size (<num_images>, <channels>, <height>, <width>)

    Example usage:
    m = Multimodel(['T1','T2'], ['DWI', 'VFlair'], {'DWI': 1.0, 'VFlair': 1.0, 'concat': 1.0}, 16, 1, True, 'max', True, True)
    m.build()
    '''
    def __init__(self, input_modalities, output_modalities, output_weights, latent_dim, img_size,
                 common_merge, ind_outs, fuse_outs):
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.latent_dim = latent_dim
        self.channels = img_size[1]
        self.common_merge = common_merge
        self.output_weights = output_weights
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs
        self.num_emb = len(input_modalities) + 1

        self.H, self.W = img_size[2], img_size[3]

    def encoder_maker(self, modality):
        filter_size = 3
        inp = Input(shape=(self.channels, self.H, self.W), name='enc_' + modality + '_input')
        conv = Conv2D(32, filter_size, padding='same', name='enc_' + modality + '_conv1')(inp) 
        act = LeakyReLU()(conv)
        conv = Conv2D(32, filter_size, padding='same', name='enc_' + modality + '_conv2')(act)
        act1 = LeakyReLU()(conv)
        # downsample 1st level
        pool = MaxPooling2D(pool_size=(2, 2))(act1)
        conv = Conv2D(64, filter_size, padding='same', name='enc_' + modality + '_conv3')(pool)
        act = LeakyReLU()(conv)
        conv = Conv2D(64, filter_size, padding='same', name='enc_' + modality + '_conv4')(act)
        act2 = LeakyReLU()(conv)

        # downsample 2nd level
        pool = MaxPooling2D(pool_size=(2, 2))(act2)
        conv = Conv2D(128, filter_size, padding='same', name='enc_' + modality + '_conv5')(pool)
        act = LeakyReLU()(conv)
        conv = Conv2D(128, filter_size, padding='same', name='enc_' + modality + '_conv6')(act)
        act = LeakyReLU()(conv)

        # upsample 2nd level
        ups = UpSampling2D(size=(2, 2))(act)
        conv = Conv2D(64, filter_size, padding='same', name='enc_' + modality + '_conv7')(ups)
        skip = concatenate([act2, conv], axis=1, name='enc_' + modality + '_skip1')
        conv = Conv2D(64, filter_size, padding='same', name='enc_' + modality + '_conv8')(skip)
        act = LeakyReLU()(conv)
        conv = Conv2D(64, filter_size, padding='same', name='enc_' + modality + '_conv9')(act)
        act = LeakyReLU()(conv)

        # upsample 2nd level
        ups = UpSampling2D(size=(2, 2))(act)
        conv = Conv2D(32, filter_size, padding='same', name='enc_' + modality + '_conv10')(ups)
        skip = concatenate([act1, conv], axis=1, name='enc_' + modality + '_skip2')
        conv = Conv2D(32, filter_size, padding='same', name='enc_' + modality + '_conv11')(skip)
        act = LeakyReLU()(conv)
        conv = Conv2D(32, filter_size, padding='same', name='enc_' + modality + '_conv12')(act)
        act = LeakyReLU()(conv)

        conv_ld = self.latent_dim
        conv = Conv2D(conv_ld, filter_size, padding='same', name='enc_' + modality + '_conv13')(act)
        lr = LeakyReLU()(conv)

        return inp, lr

    def decoder_maker(self, modality):
        print(self.latent_dim)
        filter_size = 3
        inp = Input(shape=(self.latent_dim, None, None), name='dec_' + modality + '_input') #for hemis change to self.latent_dim*2
        conv = Conv2D(32, filter_size, padding='same', activation='relu', name='dec_' + modality + '_conv1')(inp)
        conv = Conv2D(32, filter_size, padding='same', activation='relu', name='dec_' + modality + '_conv2')(conv)
        skip = concatenate([inp, conv], axis=1, name='dec_' + modality + '_skip1')
        conv = Conv2D(32, filter_size, padding='same', activation='relu', name='dec_' + modality + '_conv3')(skip)
        conv = Conv2D(32, filter_size, padding='same', activation='relu', name='dec_' + modality + '_conv4')(conv)
        skip = concatenate([skip, conv], axis=1, name='dec_' + modality + '_skip2')
        if modality == 'MASK':
            conv = Conv2D(2, 1, padding='same', name='dec_' + modality + '_conv5')(skip)
        else:
            conv = Conv2D(1, 1, padding='same', activation='relu', name='dec_' + modality + '_conv5')(skip)
        model = Model(inputs=inp, outputs=conv, name='decoder_' + modality)
        return model

    def get_embedding_distance_outputs(self, embeddings):
        if len(self.inputs) == 1:
            print ('Skipping embedding distance outputs for unimodal model')
            return []

        outputs = list()

        ind_emb = embeddings[:-1]
        weighted_rep = embeddings[-1]

        all_emb_flattened = [new_flatten(emb,name='ind_emb' + str(i)) for i,emb in enumerate(ind_emb)]
        concat_emb = concatenate(all_emb_flattened, axis=1, name='em_concat')
        #concat_emb.name = 'em_concat'

        outputs.append(concat_emb)
        #print 'making output: em_concat', concat_emb.type, concat_emb.name

        fused_emb = new_flatten(weighted_rep, name='em_fused')
        # fused_emb.name = 'em_fused'
        outputs.append(fused_emb)

        return outputs


    # HeMIS based fusion:
    # M. Havaei, N. Guizard, N. Chapados, and Y. Bengio, “HeMIS: Hetero- modal image segmentation,”
    # in MICCAI. Springer, 2016, pp. 469–477
    def hemis(self, ind_emb):
        if len(self.input_modalities) == 1:
            combined_emb1 = ind_emb[0]
            combined_emb2 = K.zeros_like(ind_emb[0])  # if we only have one input the variance is 0
        else:
            ind_emb1 = [idx for idx in ind_emb]
            combined_emb1 = merge(ind_emb1, mode=ave , name='combined_em_ave',
                                  output_shape=(self.latent_dim / 2 , None, None))
            combined_emb2 = merge(ind_emb1, mode=var, name='combined_em_var',
                                  output_shape=(self.latent_dim / 2, None, None))
        combined_emb = merge([combined_emb1, combined_emb2], mode='concat', concat_axis=1, name='combined_em',
                            output_shape=(self.latent_dim, None, None))

        new_ind_emb = []
        for i, emb in enumerate(ind_emb):
            new_ind_emb.append(merge([emb, zeros_for_var(emb)], mode='concat', concat_axis=1, name='emb_' + str(i),
                                     output_shape=(self.latent_dim, None, None)))
        ind_emb = new_ind_emb

        all_emb = ind_emb + [combined_emb]                 
        return all_emb

    def se_block(self,residual, name, ratio=8):
    #"""Contains the implementation of Squeeze-and-Excitation(SE) block.
    #As described in https://arxiv.org/abs/1709.01507.
    #"""

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name):
            channel = residual.get_shape()[-1]
        # Global average pooling
            squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)   
            assert squeeze.get_shape()[1:] == (1,1,channel)
            excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc')   
            assert excitation.get_shape()[1:] == (1,1,channel//ratio)
            excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')    
            assert excitation.get_shape()[1:] == (1,1,channel)
    # top = tf.multiply(bottom, se, name='scale')
            scale = residual * excitation     
        return scale


    def cbam_block(self, ind_emb, name, ratio=1):
    #"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    #As described in https://arxiv.org/abs/1807.06521.
    #"""
        channel_feature = self.channel(ind_emb)
        emb=[]
        for i in range(0,channel_feature.shape[1]/self.latent_dim):
            emb.append(Lambda(lambda x: x[:, i*self.latent_dim:(i+1)*self.latent_dim], name='tuple2list_'+str(i))(channel_feature))
        spatial_feature = self.spatial(emb)
        return spatial_feature

    def build(self):
        print 'Latent dimensions: ' + str(self.latent_dim)

        encoders = [self.encoder_maker(m) for m in self.input_modalities]

        ind_emb = [lr for (input, lr) in encoders]
        self.org_ind_emb = [lr for (input, lr) in encoders]
        self.inputs = [input for (input, lr) in encoders]
        

        if self.common_merge == 'hemis':
            self.all_emb = self.hemis(ind_emb)
        elif self.common_merge== 'cbam_block':
            weighted_rep = self.cbam_block(ind_emb,'hjh')
            self.all_emb = ind_emb + [weighted_rep] 
        else:
            assert self.common_merge == 'max' or self.common_merge == 'ave' or self.common_merge == 'rev_loss'
            print 'Fuse latent representations using ' + str(self.common_merge)
            if self.common_merge == 'ave': 
                weighted_rep = average(ind_emb, name='combined_em') if len(self.inputs) > 1 else ind_emb[0]
            else:
                weighted_rep = maximum(ind_emb, name='combined_em') if len(self.inputs) > 1 else ind_emb[0]
            self.all_emb = ind_emb + [weighted_rep]

        self.decoders = [self.decoder_maker(m) for m in self.output_modalities]

        outputs = get_decoder_outputs(self.output_modalities, self.decoders, self.all_emb)

        # this is for minimizing the distance between the individual embeddings
        # outputs += self.get_embedding_distance_outputs(self.all_emb)

        print 'all outputs: ', [o.name for o in outputs]

        #out_dict = {'decoder_MASK' : mae }
        # out_dict = {'em_%d_dec_%s' % (emi, dec): mae for emi in range(self.num_emb) for dec in self.output_modalities}
        out_dict = {}
        out_dict['em_0_dec_FLAIR'] = mae2
        out_dict['em_1_dec_FLAIR'] = mae2
        out_dict['em_2_dec_FLAIR'] = mae2

        out_dict['em_0_dec_MASK'] = dice_coef_loss
        out_dict['em_1_dec_MASK'] = dice_coef_loss
        out_dict['em_2_dec_MASK'] = dice_coef_loss        
        get_indiv_weight = lambda mod: self.output_weights[mod] if self.ind_outs else 0.0
        get_fused_weight = lambda mod: self.output_weights[mod] if self.fuse_outs else 0.0
        loss_weights = {}
        for dec in self.output_modalities:
            for emi in range(self.num_emb - 1):
                loss_weights['em_%d_dec_%s' % (emi, dec) ] = get_indiv_weight(dec)
            loss_weights['em_%d_dec_%s' % (self.num_emb - 1, dec) ] = get_fused_weight(dec)

        if len(self.inputs) > 1:
            if self.common_merge == 'rev_loss':
                out_dict['em_concat'] = mae
            # else:
                # out_dict['em_concat'] = embedding_distance
            # loss_weights['em_concat'] = self.output_weights['concat']

            # out_dict['em_fused'] = embedding_distance
            # loss_weights['em_fused'] = self.output_weights['concat']

        print 'output dict: ', out_dict
        print 'loss weights: ', loss_weights

        self.model = Model(input=self.inputs, output=outputs)
        self.model.summary()
        self.model.compile(optimizer=[Adam(lr=0.0001),Adam(lr=0.001)], loss=out_dict, loss_weights=loss_weights)

    def get_inputs(self, modalities):
        return [self.inputs[self.input_modalities.index(mod)] for mod in modalities]

    def get_embeddings(self, modalities):
        assert set(modalities).issubset(set(self.input_modalities))
        ind_emb = [self.all_emb[self.input_modalities.index(mod)] for mod in modalities]
        org_ind_emb = [self.org_ind_emb[self.input_modalities.index(mod)] for mod in modalities]
        print org_ind_emb[0].shape
        if self.common_merge == 'hemis':
            combined_emb1 = merge(org_ind_emb, mode=ave, name='combined_em_ave',
                                  output_shape=(self.latent_dim / 2, None, None))
            combined_emb2 = merge(org_ind_emb, mode=var, name='combined_em_var',
                                  output_shape=(self.latent_dim / 2, None, None))

            combined_emb = merge([combined_emb1, combined_emb2], mode='concat', concat_axis=1, name='combined_em',
                                 output_shape=(self.latent_dim, None, None))

            new_ind_emb = []
            for i, emb in enumerate(org_ind_emb):
                new_ind_emb.append(merge([emb, zeros_for_var(emb)], mode='concat', concat_axis=1, name='pemb_' + str(i),
                                         output_shape=(self.latent_dim, None, None)))
            ind_emb = new_ind_emb

            return ind_emb + [combined_emb]
        else:
            if len(ind_emb) > 1:
                fused_emb = merge(ind_emb, mode=self.common_merge, name='fused_em')
            else:
                fused_emb = ind_emb[0]
            return ind_emb + [fused_emb]

    def get_input(self, modality):
        assert modality in self.input_modalities
        for l in self.model.layers:
            if l.name == 'enc_' + modality + '_input':
                return l.output
        return None

    def predict_z(self, input_modalities, data, ids):
        embeddings = self.get_embeddings(input_modalities)
        inputs = [self.get_input(mod) for mod in input_modalities]
        partial_model = Model(input=inputs, output=embeddings)
        X = [data.select_for_ids(inmod, ids) for inmod in input_modalities]
        Z = partial_model.predict(X)
        assert len(Z) == len(embeddings)
        return Z

    def spatial(self,ind_emb):
        kernel_size = 3
        kernel_initializer1 =tf.contrib.layers.variance_scaling_initializer()
        # with tf.variable_scope(name):
        avg_emb = []
        max_emb = []
        std_emb = []
        for emb in ind_emb:
            avg_pool = Lambda(lambda x: K.mean(x, axis=[1], keepdims=True))(emb)
            assert avg_pool.get_shape()[1] == 1
            avg_emb.append(avg_pool)
            max_pool = Lambda(lambda x: K.max(x, axis=[1], keepdims=True))(emb)
            assert max_pool.get_shape()[1] == 1
            max_emb.append(max_pool)
            # std_pool = Lambda(lambda x: K.std(x, axis=[1], keepdims=True))(emb)
            # assert std_pool.get_shape()[1] == 1
            # std_emb.append(std_pool)   

        avg_emb = Concatenate(axis=1)(avg_emb)
        max_emb = Concatenate(axis=1)(max_emb)
        # std_emb = Concatenate(axis=1)(std_emb)
        concat = Concatenate(axis=1)([avg_emb,max_emb])
        #assert concat.get_shape()[1] == 6
        concat = Lambda(lambda x: K.expand_dims(x,0))(concat)
        concat = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3,4)))(concat)
        concat = Conv3D(filters=1,
                            kernel_size=[kernel_size,kernel_size,kernel_size],
                            strides=[1,1,1],
                            padding="same",
                            activation=None,
                            kernel_initializer='random_uniform',
                            use_bias=False,
                            name='conv')(concat)
        #assert concat.get_shape()[-1] == 1
        concat = Activation('sigmoid')(concat)
        concat = Lambda(lambda x: K.squeeze(x,0))(concat)

        ind_emb = [Multiply()([a,Lambda(lambda x: K.permute_dimensions(x,(1,0,2,3)))(concat)]) for a in ind_emb]
        # concat = Lambda(lambda x: K.permute_dimensions(x,(1,0,2,3)))(concat)
        # return Lambda(lambda x: x[:,0:4,:,:])(concat)
        return Lambda(lambda x: ave(x))(ind_emb)

    def channel(self,ind_emb):

        #kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        #bias_initializer = tf.constant_initializer(value=0.0)
        ratio = 4
        # with tf.variable_scope(name):
        scale = []
        output =[]
        channel = ind_emb[0].get_shape()[1]
        shared_mean_1 = Dense(units=int(channel//ratio),
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='mean_1')
        shared_mean_2 = Dense(units=int(channel),                             
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='mean_2')
        shared_max_1 = Dense(units=int(channel//ratio),
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='max_1')
        shared_max_2 = Dense(units=int(channel),                             
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='max_2')
        shared_std_1 = Dense(units=int(channel//ratio),
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='std_1')
        shared_std_2 = Dense(units=int(channel),                             
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            name='std_2')        
                                             
        for i,emb in enumerate(ind_emb):
            #emb = K.permute_dimensions(emb, (0, 2, 3,1))
            avg_pool = Lambda(lambda x: K.mean(x, axis=[0,2,3]))(emb)
            avg_pool = Lambda(lambda x: K.expand_dims(x, axis=0))(avg_pool)
                    
            assert avg_pool.get_shape()[-1] == channel
            avg_pool = shared_mean_1(avg_pool)   
            assert avg_pool.get_shape()[-1] == channel//ratio
            avg_pool = shared_mean_2(avg_pool)    
            assert avg_pool.get_shape()[-1] == channel

            max_pool = Lambda(lambda x: K.max(x, axis=[0,2,3]))(emb)
            max_pool = Lambda(lambda x: K.expand_dims(x, axis=0))(max_pool)    
            assert max_pool.get_shape()[-1] == channel
            max_pool = shared_max_1(max_pool)   
            assert max_pool.get_shape()[-1] == channel//ratio
            max_pool = shared_max_2(max_pool)  
            assert max_pool.get_shape()[-1] == channel

            std_pool = Lambda(lambda x: K.std(x, axis=[0,2,3]))(emb)
            std_pool = Lambda(lambda x: K.expand_dims(x,axis=0))(std_pool)    
            assert std_pool.get_shape()[-1] == channel
            std_pool = shared_std_1(std_pool)
            assert std_pool.get_shape()[-1] == channel//ratio
            std_pool = shared_std_2(std_pool)  
            assert std_pool.get_shape()[-1] == channel

            added = Add()([avg_pool, max_pool, std_pool])
            activation = Activation('sigmoid')(added)
            # scale = Lambda(lambda x: K.permute_dimensions(x,(1,0)))(activation)
            # scale = Concatenate(axis=-1)([scale for j in range(channel)])
            if i == 0:
                # output = K.permute_dimensions(K.dot(K.permute_dimensions(emb, (0, 2, 3,1)),scale),(0,3,2,1))
                emb = Lambda(lambda x: K.permute_dimensions(x,(0, 2, 3,1)))(emb)
                mul = Multiply()([emb, activation])
                output = Lambda(lambda x: K.permute_dimensions(x,(0,3,1,2)))(mul)
            else:
                # output = K.concatenate([output,K.permute_dimensions(K.dot(K.permute_dimensions(emb, (0, 2, 3,1)),scale),(0,3,2,1))],axis=1)
                emb = Lambda(lambda x: K.permute_dimensions(x,(0, 2, 3,1)))(emb)
                mul = Multiply()([emb, activation])
                mul = Lambda(lambda x: K.permute_dimensions(x,(0,3,1,2)))(mul) 
                output = Concatenate(axis=1)([output, mul])               
        # emb = Lambda(lambda x: K.permute_dimensions(x,(0, 2, 3,1)))(ind_emb[0])
        return output

def get_decoder_outputs(output_modalities, decoders, embeddings):
    assert len(output_modalities) == len(decoders)

    outputs = list()
    for di, decode in enumerate(decoders):
        for emi, em in enumerate(embeddings):
            out_em = decode(em)
            name = 'em_' + str(emi) + '_dec_' + output_modalities[di]
            l = Lambda(lambda x: x + 0, name=name)(out_em)
            #l.name = name
            outputs.append(l)
            #print 'making output:', em.type, out_em.type, name
    return outputs

def embedding_distance(y_true,y_pred):
    return K.var(y_pred, axis=1)

def new_flatten(emb, name=''):
    l = Lambda(lambda x: K.batch_flatten(x), name='pre' + name)(emb)
    l = Lambda(lambda x: K.expand_dims(x, axis=1), name=name)(l)
    return l

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))
def mae2(y_true, y_pred):
    return 1000*K.mean(K.abs(y_pred - y_true))
def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 
def challenge_mae_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2)[...,1:])
    # y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.permute_dimensions(y_pred,(0,2,3,1))[...,1:])
    return K.mean(K.sum(K.abs(y_pred_f - y_true_f), axis=-1))

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (1000*true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2)[...,1:])
    # y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.permute_dimensions(y_pred,(0,2,3,1))[...,1:])
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2.0*intersection + smooth) / (K.sum(K.square(y_true_f),-1) + K.sum(K.square(y_pred_f),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 100*(1-dice_coef(y_true, y_pred))

def var(embeddings):
    emb = embeddings[0]
    shape = (emb.shape[1], emb.shape[2], emb.shape[3])
    sz = shape[0] * shape[1] * shape[2]
    flat_embs = [K.reshape(emb, (tf.shape(emb)[0], 1, sz)) for emb in embeddings]
    emb_var = K.var(K.concatenate(flat_embs, axis=1), axis=1, keepdims=True)
    return K.reshape(emb_var, tf.shape(embeddings[0]))

def ave(embeddings):
    emb = embeddings[0]
    shape = (emb.shape[1], emb.shape[2], emb.shape[3])
    sz = shape[0] * shape[1] * shape[2]
    flat_embs = [K.reshape(emb, (tf.shape(emb)[0], 1, sz)) for emb in embeddings]
    emb_ave = K.mean(K.concatenate(flat_embs, axis=1), axis=1, keepdims=True)
    return K.reshape(emb_ave, tf.shape(embeddings[0]))

def zeros_for_var(emb):
    l = Lambda(lambda x: K.zeros_like(x))(emb)
    return l