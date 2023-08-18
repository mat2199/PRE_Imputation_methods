

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf

# For MIDA

import MIDASpy as md

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

from MIWAE import *
from missing_mechanism import *

def missing_method(raw_data, mechanism='mcar',t=0.2) :
    
    data = raw_data.copy()
    rows, cols = data.shape
    
    # missingness threshold
    
    if mechanism == 'mcar' :
        # uniform random vector
        v = np.random.uniform(size=(rows, cols))

        # missing values where v<=t 
        # mask is a matrix where mij equals to True if vij<=t and False otherwise
        mask = (v<=t)
        data[mask] = np.nan


        
    
    elif mechanism == 'mar' :
        nump=raw_data.to_numpy()
        mask=MAR_mask(X=nump,p=t,p_obs=0.0)
        data[mask]=np.nan

    elif mechanism=="mnar":
        nump=raw_data.to_numpy()
        mask=MNAR_self_mask_logistic(nump,p=t)
        data[mask]=np.nan
        
    else :
        print("Error : There is no such mechanism")
        raise
        
    return data, ~mask

    
def comparison_std(full_data,missing_mecha,prop_NA,comp_methods,nb_simu,M,device,Mida_param=None,Miwae_param=None,Rf_param=None,Miwae_std_param=None,save=False):
    xcomp=(full_data-np.mean(full_data,0))/np.std(full_data,0)
    nb_methods=len(comp_methods)
    nb_perc=len(prop_NA)
    index=[method for method in comp_methods]
    col=[" ".join((str(prop*100),"%")) for prop in prop_NA]
    tmp_res=np.zeros((nb_methods,nb_perc))
    tmp_bp=[[ ['0' for col in range(nb_simu)] for col in range(nb_perc)] for row in range(nb_methods)]
    bp_df=pd.DataFrame(data=tmp_bp,columns=col,index=index)
    res_df=pd.DataFrame(data=tmp_res,columns=col,index=index)
    res={}
    boxplot={}
    for mecha in missing_mecha:
        print(mecha.upper())
        key=mecha.upper()
        res[key]=res_df.copy()
        boxplot[key]=bp_df.copy()
        for prop in prop_NA:
            
            print(" ".join(("Computation for",str(prop*100),"% of NA values")))
            mse_miwae=[]
            mse_mida=[]
            mse_rf=[]
            mse_mean=[]
            mse_miwae_std=[]
            for i in range(1,nb_simu+1):
                seed=i
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.seed()
                tf.random.set_seed(seed)
                miss_tmp,mask=missing_method(raw_data=full_data,mechanism=mecha,t=prop)
                miss=(miss_tmp-np.nanmean(miss_tmp,0))/np.nanstd(miss_tmp,0)
                if "MIDA" in comp_methods:                             
                    Midas_imputer=md.Midas(layer_structure=Mida_param[0],learn_rate=Mida_param[1],input_drop=Mida_param[2],train_batch=Mida_param[3],seed=Mida_param[4])
                    miss_mid=miss.copy()
                    Midas_imputer.build_model(imputation_target=miss_mid,verbose=False)
                    Midas_imputer.train_model(training_epochs=Mida_param[5],verbose=False)
                    Mida_imputations=Midas_imputer.generate_samples(m=M,verbose=False).output_list
                    tmp_mse_mida=avg_mse(list_imp=Mida_imputations,xtrue=xcomp,mask=mask)
                    mse_mida.append(tmp_mse_mida)
                if "MIWAE" in comp_methods:
                    n,p=miss.shape
                    miss_miw=miss.copy()
                    p_z=prior(d=Miwae_param[0],device=device)
                    encoder,decoder=build_encoder_decoder(p,h=Miwae_param[1],d=Miwae_param[0])
                    optimizer=build_optimizer(encoder=encoder,decoder=decoder)
                    m=train_MIWAE(encoder=encoder,decoder=decoder,optimizer=optimizer,d=Miwae_param[0],p_z=p_z,miss_data=miss_miw,raw_data=xcomp,device=device,K=Miwae_param[2],verbose=False,n_epochs=Miwae_param[3])
                    Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=Miwae_param[0],p=p,p_z=p_z,device=device,M=M)
                    tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=xcomp,mask=mask)
                    mse_miwae.append(tmp_mse_miwae)
                if "MIWAE_standardization" in comp_methods:
                    n,p=miss.shape
                    miss_miw_std=miss.copy()
                    p_z_std=prior(d=Miwae_std_param[0],device=device)
                    encoder_std,decoder_std=build_encoder_decoder(p,h=Miwae_std_param[1],d=Miwae_std_param[0])
                    optimizer_std=build_optimizer(encoder=encoder_std,decoder=decoder_std)
                    m=train_MIWAE_standardization(encoder=encoder_std,decoder=decoder_std,optimizer=optimizer_std,d=Miwae_std_param[0],p_z=p_z_std,miss_data=miss_miw_std,mean_0=np.nanmean(miss,0),std_0=np.nanstd(miss,0),raw_data=xcomp,device=device,K=Miwae_std_param[2],verbose=False,n_epochs=Miwae_std_param[3])
                    Miwae_std_imputations=miwae_multiple_impute(data=miss_miw_std,encoder=encoder_std,decoder=decoder_std,d=Miwae_std_param[0],p=p,p_z=p_z_std,device=device,M=M)
                    tmp_mse_miwae_std=avg_mse(list_imp=Miwae_std_imputations,xtrue=xcomp,mask=mask)
                    mse_miwae_std.append(tmp_mse_miwae_std)
                if "RF" in comp_methods:
                    missforest = IterativeImputer(max_iter=Rf_param[0], estimator=ExtraTreesRegressor(n_estimators=Rf_param[1]))
                    missf=miss.copy()
                    missforest.fit(missf)
                    xhat_mf = missforest.transform(missf)
                    tmp_mse_rf=mse(xhat=xhat_mf,xtrue=xcomp,mask=mask)
                    mse_rf.append(tmp_mse_rf)
                if "MEAN" in comp_methods:
                    mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                    missm=miss.copy()
                    mean_imp.fit(missm)
                    xhat_mean = mean_imp.transform(missm)
                    tmp_mse_mean=mse(xhat=xhat_mean,xtrue=xcomp,mask=mask)
                    mse_mean.append(tmp_mse_mean)
                
                if i==nb_simu//2:
                    print(" ".join(["Simulation 1 -",str(nb_simu//2),"done"]))
                if i==nb_simu:
                    print(" ".join(["Simulation",str(nb_simu//2+1),"-",str(nb_simu),"done"]))
 
                
            if "MIDA" in comp_methods:
                boxplot[key].loc["MIDA"," ".join((str(prop*100),"%"))]=mse_mida
                res[key].loc["MIDA"," ".join((str(prop*100),"%"))]=np.mean(mse_mida)
            if "MIWAE" in comp_methods:
                boxplot[key].loc["MIWAE"," ".join((str(prop*100),"%"))]=mse_miwae
                res[key].loc["MIWAE"," ".join((str(prop*100),"%"))]=np.mean(mse_miwae)
            if "MIWAE_standardization" in comp_methods:
                boxplot[key].loc["MIWAE_standardization"," ".join((str(prop*100),"%"))]=mse_miwae_std
                res[key].loc["MIWAE_standardization"," ".join((str(prop*100),"%"))]=np.mean(mse_miwae_std)
            if "RF" in comp_methods:
                boxplot[key].loc["RF"," ".join((str(prop*100),"%"))]=mse_rf
                res[key].loc["RF"," ".join((str(prop*100),"%"))]=np.mean(mse_rf)
            if "MEAN" in comp_methods:
                boxplot[key].loc["MEAN"," ".join((str(prop*100),"%"))]=mse_mean
                res[key].loc["MEAN"," ".join((str(prop*100),"%"))]=np.mean(mse_mean)
        if save:
            directory = os.getcwd()
            file_name="\ ".join([directory,"_".join([key,"results.csv"])])
            res[key].to_csv(file_name)
            

    return res,boxplot

    
def comparison(full_data,missing_mecha,prop_NA,comp_methods,nb_simu,M,device,Mida_param=None,Miwae_param=None,Rf_param=None,Miwae_std_param=None,save=False):
    nb_methods=len(comp_methods)
    nb_perc=len(prop_NA)
    index=[method for method in comp_methods]
    col=[" ".join((str(prop*100),"%")) for prop in prop_NA]
    tmp_res=np.zeros((nb_methods,nb_perc))
    tmp_bp=[[ ['0' for col in range(nb_simu)] for col in range(nb_perc)] for row in range(nb_methods)]
    bp_df=pd.DataFrame(data=tmp_bp,columns=col,index=index)
    res_df=pd.DataFrame(data=tmp_res,columns=col,index=index)
    res={}
    boxplot={}
    for mecha in missing_mecha:
        print(mecha.upper())
        key=mecha.upper()
        res[key]=res_df.copy()
        boxplot[key]=bp_df.copy()
        for prop in prop_NA:
            
            print(" ".join(("Computation for",str(prop*100),"% of NA values")))
            mse_miwae=[]
            mse_mida=[]
            mse_rf=[]
            mse_mean=[]
            mse_miwae_std=[]
            for i in range(1,nb_simu+1):
                seed=i
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.seed()
                tf.random.set_seed(seed)
                miss,mask=missing_method(raw_data=full_data,mechanism=mecha,t=prop)
                if "MIDA" in comp_methods:                             
                    Midas_imputer=md.Midas(layer_structure=Mida_param[0],learn_rate=Mida_param[1],input_drop=Mida_param[2],train_batch=Mida_param[3],seed=Mida_param[4])
                    miss_mid=miss.copy()
                    Midas_imputer.build_model(imputation_target=miss_mid,verbose=False)
                    Midas_imputer.train_model(training_epochs=Mida_param[5],verbose=False)
                    Mida_imputations=Midas_imputer.generate_samples(m=M,verbose=False).output_list
                    tmp_mse_mida=avg_mse(list_imp=Mida_imputations,xtrue=full_data,mask=mask)
                    mse_mida.append(tmp_mse_mida)
                if "MIWAE" in comp_methods:
                    n,p=miss.shape
                    miss_miw=miss.copy()
                    p_z=prior(d=Miwae_param[0],device=device)
                    encoder,decoder=build_encoder_decoder(p,h=Miwae_param[1],d=Miwae_param[0])
                    optimizer=build_optimizer(encoder=encoder,decoder=decoder)
                    m=train_MIWAE(encoder=encoder,decoder=decoder,optimizer=optimizer,d=Miwae_param[0],p_z=p_z,miss_data=miss_miw,raw_data=full_data,device=device,K=Miwae_param[2],verbose=False,n_epochs=Miwae_param[3])
                    Miwae_imputations=miwae_multiple_impute(data=miss_miw,encoder=encoder,decoder=decoder,d=Miwae_param[0],p=p,p_z=p_z,device=device,M=M)
                    tmp_mse_miwae=avg_mse(list_imp=Miwae_imputations,xtrue=full_data,mask=mask)
                    mse_miwae.append(tmp_mse_miwae)
                if "MIWAE_standardization" in comp_methods:
                    n,p=miss.shape
                    miss_miw_std=miss.copy()
                    p_z_std=prior(d=Miwae_std_param[0],device=device)
                    encoder_std,decoder_std=build_encoder_decoder(p,h=Miwae_std_param[1],d=Miwae_std_param[0])
                    optimizer_std=build_optimizer(encoder=encoder_std,decoder=decoder_std)
                    m=train_MIWAE_standardization(encoder=encoder_std,decoder=decoder_std,optimizer=optimizer_std,d=Miwae_std_param[0],p_z=p_z_std,miss_data=miss_miw_std,mean_0=np.nanmean(miss,0),std_0=np.nanstd(miss,0),raw_data=full_data,device=device,K=Miwae_std_param[2],verbose=False,n_epochs=Miwae_std_param[3])
                    Miwae_std_imputations=miwae_multiple_impute(data=miss_miw_std,encoder=encoder_std,decoder=decoder_std,d=Miwae_std_param[0],p=p,p_z=p_z_std,device=device,M=M)
                    tmp_mse_miwae_std=avg_mse(list_imp=Miwae_std_imputations,xtrue=full_data,mask=mask)
                    mse_miwae_std.append(tmp_mse_miwae_std)
                if "RF" in comp_methods:
                    missforest = IterativeImputer(max_iter=Rf_param[0], estimator=ExtraTreesRegressor(n_estimators=Rf_param[1]))
                    missf=miss.copy()
                    missforest.fit(missf)
                    xhat_mf = missforest.transform(missf)
                    tmp_mse_rf=mse(xhat=xhat_mf,xtrue=full_data,mask=mask)
                    mse_rf.append(tmp_mse_rf)
                if "MEAN" in comp_methods:
                    mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                    missm=miss.copy()
                    mean_imp.fit(missm)
                    xhat_mean = mean_imp.transform(missm)
                    tmp_mse_mean=mse(xhat=xhat_mean,xtrue=full_data,mask=mask)
                    mse_mean.append(tmp_mse_mean)
                
                if i==nb_simu//2:
                    print(" ".join(["Simulation 1 -",str(nb_simu//2),"done"]))
                if i==nb_simu:
                    print(" ".join(["Simulation",str(nb_simu//2+1),"-",str(nb_simu),"done"]))

 
                
            if "MIDA" in comp_methods:
                boxplot[key].loc["MIDA"," ".join((str(prop*100),"%"))]=mse_mida
                res[key].loc["MIDA"," ".join((str(prop*100),"%"))]=np.mean(mse_mida)
            if "MIWAE" in comp_methods:
                boxplot[key].loc["MIWAE"," ".join((str(prop*100),"%"))]=mse_miwae
                res[key].loc["MIWAE"," ".join((str(prop*100),"%"))]=np.mean(mse_miwae)
            if "MIWAE_standardization" in comp_methods:
                boxplot[key].loc["MIWAE_standardization"," ".join((str(prop*100),"%"))]=mse_miwae_std
                res[key].loc["MIWAE_standardization"," ".join((str(prop*100),"%"))]=np.mean(mse_miwae_std)
            if "RF" in comp_methods:
                boxplot[key].loc["RF"," ".join((str(prop*100),"%"))]=mse_rf
                res[key].loc["RF"," ".join((str(prop*100),"%"))]=np.mean(mse_rf)
            if "MEAN" in comp_methods:
                boxplot[key].loc["MEAN"," ".join((str(prop*100),"%"))]=mse_mean
                res[key].loc["MEAN"," ".join((str(prop*100),"%"))]=np.mean(mse_mean)
        if save:
            directory = os.getcwd()
            file_name="\ ".join([directory,"_".join([key,"results.csv"])])
            file_name_2="\ ".join([directory,"_".join([key,"boxplot_results.csv"])])
            res[key].to_csv(file_name)
            boxplot[key].to_csv(file_name_2)
            

    return res,boxplot





        

        
        
    
        

        
        
    


            