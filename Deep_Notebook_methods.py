from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer

import torch
import torchvision
import torch.nn as nn
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import torch.distributions as td

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cpu")

################################################
### Functions for the denoising autoencoders ###
################################################

###Function useful to manage the data for an autoencoder. 
###See TensorFlow documentation: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

###Function created by Aurelien Geron: https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
        shape=[self.dense.input_shape[-1]],
        initializer="zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

###Function that creates the layers for SDAI spt
def layers_AE(layers_dense, activ):
    dense = []
    for i, j  in zip(layers_dense,[activ]*len(layers_dense)):
        dense.append(keras.layers.Dense(i, activation = j))
    return dense

###Function that does pre training for and done by Aurelien Geron:
###https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb
def pre_train_layer(X_train, X_valid, loss, optimizer, dense, mask, n_epochs=10, output_activation=None, metrics=None, num_layer = 0, miss_dataset=None, valid_dataset=None, batch_size=None):
    
    n_inputs = X_train.shape[1]
    dense_init = keras.layers.Input(shape=[n_inputs])
    
    encoder = keras.models.Sequential([
        dense[num_layer],
    ])
    
    decoder = keras.models.Sequential([
        DenseTranspose(dense[num_layer], activation = output_activation),
    ])
    
    autoencoder = keras.models.Sequential([dense_init,encoder, decoder])
    
    if num_layer == 0:
        train_all(autoencoder, n_epochs, optimizer, mask, miss_dataset, valid_dataset, batch_size)
    else:
        autoencoder.compile(optimizer, loss = loss , metrics=metrics)
        autoencoder.fit(X_train, X_train, epochs=n_epochs, validation_data=[X_valid, X_valid])
        
    return encoder, decoder, encoder(X_train), encoder(X_valid)

###Function that makes all pre-triaining
def pre_training(X_t, X_v, num_epochs_pt, output_activ, layers_dense, activ, miss_dataset, valid_dataset, batch_size, opt, mask):
    enc_list = []
    dec_list = []
    dense = layers_AE(layers_dense, activ)
    enc, dec, X_t, X_v = pre_train_layer(X_t, X_v, loss = loss_custom, optimizer = opt, dense = dense, mask=mask, output_activation = output_activ, num_layer = 0, n_epochs=num_epochs_pt, miss_dataset=miss_dataset, valid_dataset=valid_dataset, batch_size=batch_size)
    enc_list.append(enc)
    dec_list.append(dec)
    
    for n in range(1,len(layers_dense),1):
        enc, dec, X_t, X_v = pre_train_layer(X_t, X_v, loss= loss_custom, optimizer = opt, dense = dense, mask=mask, output_activation = factiv, num_layer = n, n_epochs=num_epochs_pt, miss_dataset=miss_dataset, valid_dataset=valid_dataset, batch_size=batch_size)
        enc_list.append(enc)
        dec_list.append(dec)
        
    return enc_list, dec_list, dense

###Loss function for SDAI pt
def loss_custom(targets, outputs):
    loss = tf.sqrt(tf.reduce_mean(tf.square(targets - outputs)))
    return loss

###Function that makes a pre-training step
def train_step(net, example, validation, optimizer, mask_b):
    with tf.GradientTape() as tape:
        output = net(example)
        output_valid = net(validation)
        loss_val = tf.sqrt(tf.reduce_mean(tf.square(output_valid - validation)))
        loss = tf.sqrt(tf.reduce_mean(tf.square(output - example)[mask_b]))
        variables = net.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return loss, loss_val

###Function that makes the complete training
def train_all(net,nb_epochs, opt, mask, miss_dataset, valid_dataset, batch_size):
    loss_epochs = []
    loss_val_epochs = []
    epoch = 1
    mean = 1
    while epoch<=nb_epochs and mean>0.0005:
        loss_batch = []
        loss_val_batch = []
        i = 0
        for m, v in zip(miss_dataset,valid_dataset):###parcours des batchs
            example = m[0]
            inf = i * batch_size
            print(inf)
            sup = inf + m[0].shape[0]
            mask_b = mask[inf:sup,:]
            validation = v[0]
            lb, lvb = train_step(net, example, validation, opt, mask_b)
            loss_batch.append(lb)
            loss_val_batch.append(lvb)
            i = i + 1
        loss_epochs.append(np.mean(loss_batch))
        loss_val_epochs.append(np.mean(loss_val_batch))
        if epoch <4:
            mean = 1
        else:
            v = loss_epochs[len(loss_epochs)-4:]
            mean = np.mean([v[0]-v[1], v[1]-v[2], v[2]-v[3]])

        epoch = epoch + 1
    return  (loss_epochs, loss_val_epochs)


#########################################################
### Functions for the variational autoencoder (MIWAE) ###
#########################################################

### Loss function for the algorithm MIWAE
def miwae_loss(iota_x,mask,d,K,p_z,encoder,decoder):
    batch_size = iota_x.shape[0]
    nb_var = iota_x.shape[1]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :nb_var]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., nb_var:(2*nb_var)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*nb_var):(3*nb_var)]) + 3

    data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask,[K,1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,nb_var])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

    return neg_bound

### Function that imputes missing values
def miwae_impute(iota_x,mask,L,d,p_z,encoder,decoder):
    batch_size = iota_x.shape[0]
    nb_var = iota_x.shape[1]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :nb_var]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., nb_var:(2*nb_var)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*nb_var):(3*nb_var)]) + 3

    data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).to(device)
    tiledmask = torch.Tensor.repeat(mask,[L,1]).to(device)

    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,nb_var])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.sample().reshape([L,batch_size,nb_var])
    xm=torch.einsum('ki,kij->ij', imp_weights, xms) 

    return xm


# Function that initializes the weights
def weights_init(layer):
    if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)

# RMSE
def rmse(xhat,xtrue,mask): 
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.sqrt(np.mean(np.power(xhat-xtrue,2)[~mask]))
