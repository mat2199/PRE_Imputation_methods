
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os

class TensorFlowLogger:
    def __init__(self):
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(logging.FileHandler(os.path.join(os.getcwd(), 'tensorflow.log')))

    def set_log_level(self, level):
        self.logger.setLevel(level)

    def get_log_level(self):
        return self.logger.level

    def log(self, message, level=logging.INFO):
        if self.logger.isEnabledFor(level):
            self.logger.log(level, message)

logger = TensorFlowLogger()
logger.set_log_level(logging.ERROR)

import tensorflow as tf

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import MinMaxScaler


# For MIDA

import MIDASpy as md


# For MIWAE

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

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from MIWAE import *




def optimize_MIDA_param(lr_array,full_data,miss_data,mask,plot=True):
    mse=[]
    for lr in np.nditer(lr_array):
        Midas_imputer=md.Midas(layer_structure=[64,64,64],learn_rate=lr,input_drop=0.8,train_batch=32,seed=656);
        miss_mida=miss_data.copy()
        Midas_imputer.build_model(imputation_target=miss_mida,verbose=False);
        Midas_imputer.train_model(training_epochs=100,verbose=False);
        Mida_imputations=Midas_imputer.generate_samples(m=1000,verbose=False).output_list;
        tmp_mse_mida=avg_mse(list_imp=Mida_imputations,xtrue=full_data,mask=mask)
        mse.append(tmp_mse_mida)
    print("The minimum mse value is obtained for a learning rate : {lr}".format(lr=np.around(lr_array[mse.index(min(mse))],4)))
    if plot :
        plt.plot(lr_array,mse)
        plt.xlabel("Learning rate")
        plt.ylabel("MSE")
        plt.title("MSE for each value of learning rate")




def optimize_MIWAE_param(d_array,K_array,full_data,miss_data,mask,plot=True,device=device):
    mse_d=[]
    mse_K=[]
    for d_tmp in np.nditer(d_array):
        d=int(d_tmp)
        n,p=miss_data.shape
        miss_miw=miss_data.copy()
        p_z=prior(d=d,device=device)
        encoder,decoder=build_encoder_decoder(p,h=64,d=d)
        optimizer=build_optimizer(encoder=encoder,decoder=decoder)
        m=train_MIWAE(encoder=encoder,decoder=decoder,optimizer=optimizer,d=d,p_z=p_z,miss_data=miss_miw,raw_data=full_data,device=device,K=100,verbose=False,n_epochs=100)
        Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=d,p=p,p_z=p_z,device=device,M=1000)
        tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=full_data,mask=mask)
        mse_d.append(tmp_mse_miwae)
    d_opt_tmp=d_array[mse_d.index(min(mse_d))]
    d_opt=int(d_opt_tmp)
    for K_tmp in np.nditer(K_array):
        K=int(K_tmp)
        n,p=miss_data.shape
        miss_miw=miss_data.copy()
        p_z=prior(d=d_opt,device=device)
        encoder,decoder=build_encoder_decoder(p,h=64,d=d_opt)
        optimizer=build_optimizer(encoder=encoder,decoder=decoder)
        m=train_MIWAE(encoder=encoder,decoder=decoder,optimizer=optimizer,d=d_opt,p_z=p_z,miss_data=miss_miw,raw_data=full_data,device=device,K=K,verbose=False,n_epochs=100)
        Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=d_opt,p=p,p_z=p_z,device=device,M=1000)
        tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=full_data,mask=mask)
        mse_K.append(tmp_mse_miwae)
    K_opt=K_array[mse_K.index(min(mse_K))]
    print("The minimum mse value is obtained for a dimension of the latent space d: {d} and a number of iteration for the importance sampling of the MIWAE bound construction K :{K}".format(d=d_opt,K=K_opt))
    if plot :
        f,(ax1,ax2)=plt.subplots(2)
        plt.subplots_adjust(hspace=0.6)
        ax1.plot(d_array,mse_d)
        ax2.plot(K_array,mse_K)
        ax1.set_xlabel("d")
        ax1.set_ylabel("MSE")
        ax2.set_xlabel("K")
        ax2.set_ylabel("MSE")
        ax1.set_title("MSE for each value of the dimension of the latent space,K fixed at 100")
        ax2.set_title("MSE for each value of iteration in the MIWAE bound importance sampling, d fixed at {d}".format(d=d_opt))


def optimize_MIWAE_std_param(d_array,K_array,full_data,miss_data,mask,plot=True,device=device):
    mse_d=[]
    mse_K=[]
    for d_tmp in np.nditer(d_array):
        d=int(d_tmp)
        n,p=miss_data.shape
        miss_miw=miss_data.copy()
        p_z=prior(d=d,device=device)
        encoder,decoder=build_encoder_decoder(p,h=64,d=d)
        optimizer=build_optimizer(encoder=encoder,decoder=decoder)
        m=train_MIWAE_standardization(encoder=encoder,decoder=decoder,optimizer=optimizer,d=d,p_z=p_z,mean_0=np.mean(full_data,0),std_0=np.std(full_data,0),miss_data=miss_miw,raw_data=full_data,device=device,K=100,verbose=False,n_epochs=100)
        Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=d,p=p,p_z=p_z,device=device,M=1000)
        tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=full_data,mask=mask)
        mse_d.append(tmp_mse_miwae)
    d_opt_tmp=d_array[mse_d.index(min(mse_d))]
    d_opt=int(d_opt_tmp)
    for K_tmp in np.nditer(K_array):
        K=int(K_tmp)
        n,p=miss_data.shape
        miss_miw=miss_data.copy()
        p_z=prior(d=d_opt,device=device)
        encoder,decoder=build_encoder_decoder(p,h=64,d=d_opt)
        optimizer=build_optimizer(encoder=encoder,decoder=decoder)
        m=train_MIWAE_standardization(encoder=encoder,decoder=decoder,optimizer=optimizer,d=d_opt,mean_0=np.mean(full_data,0),std_0=np.std(full_data,0),p_z=p_z,miss_data=miss_miw,raw_data=full_data,device=device,K=K,verbose=False,n_epochs=100)
        Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=d_opt,p=p,p_z=p_z,device=device,M=1000)
        tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=full_data,mask=mask)
        mse_K.append(tmp_mse_miwae)
    K_opt=K_array[mse_K.index(min(mse_K))]
    print("The minimum mse value is obtained for a dimension of the latent space d: {d} and a number of iteration for the importance sampling of the MIWAE bound construction K :{K}".format(d=d_opt,K=K_opt))
    if plot :
        f,(ax1,ax2)=plt.subplots(2)
        plt.subplots_adjust(hspace=0.6)
        ax1.plot(d_array,mse_d)
        ax2.plot(K_array,mse_K)
        ax1.set_xlabel("d")
        ax1.set_ylabel("MSE")
        ax2.set_xlabel("K")
        ax2.set_ylabel("MSE")
        ax1.set_title("MSE for each value of the dimension of the latent space,K fixed at 100")
        ax2.set_title("MSE for each value of iteration in the MIWAE with standardization bound importance sampling, d fixed at {d}".format(d=d_opt))


def optimize_RF_param(n_est_array,full_data,miss_data,mask,plot=True):
    mse_RF=[]
    for n in np.nditer(n_est_array):
        missforest = IterativeImputer(max_iter=10, estimator=ExtraTreesRegressor(n_estimators=int(n)))
        missf=miss_data.copy()
        missforest.fit(missf)
        xhat_mf = missforest.transform(missf)
        tmp_mse_rf=mse(xhat=xhat_mf,xtrue=full_data,mask=mask)
        mse_RF.append(tmp_mse_rf)
    n_opt=n_est_array[mse_RF.index(min(mse_RF))]
    print("The minimum mse value is obtained for a number of trees in the forest of : {n}".format(n=n_opt))
    if plot :
        plt.plot(n_est_array,mse_RF)
        plt.xlabel("Number of trees in the forest")
        plt.ylabel("MSE")
        plt.title("MSE for each value of trees in the forest")