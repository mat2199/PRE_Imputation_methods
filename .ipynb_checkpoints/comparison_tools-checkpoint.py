

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


def plot_result(results,comp_meth,save=False,name=None):
    mechanism_results=list(results)
    nb_mech=len(results)
    if nb_mech==1:
        mse_MIDA=None
        mse_MIWAE=None
        mse_RF=None
        mse_MEAN=None
        mse_MIWAE_std=None
        legend=[]
        mecha=mechanism_results[0]
        mse_data=results[mechanism_results[0]]
        for method in comp_meth:
            if method=="MIDA":
                mse_MIDA=mse_data.loc["MIDA"]
            elif method=="MIWAE":
                mse_MIWAE=mse_data.loc["MIWAE"]
            elif method=="MIWAE_standardization":
                mse_MIWAE_std=mse_data.loc["MIWAE_standardization"]
            elif method=="RF":
                mse_RF=mse_data.loc["RF"]
            elif method=="MEAN":
                mse_MEAN=mse_data.loc["MEAN"]
        f,ax=plt.subplots(1,constrained_layout)
        if mse_MIDA is not None:
            ax.plot(mse_MIDA,color="red")
            legend+=["MIDA"]
        if mse_MIWAE is not None:
            ax.plot(mse_MIWAE,color="blue")
            legend+=["MIWAE"]
        if mse_MIWAE_std is not None:
            ax.plot(mse_MIWAE_std,color="orchid")
            legend+=["MIWAE_standardization"]
        if mse_RF is not None:
            ax.plot(mse_RF,color="green")
            legend+=["MissForest"]
        if mse_MEAN is not None:
            ax.plot(mse_MEAN,color="grey")
            legend+=["MEAN"]
        ax.set_title(" ".join((mecha,"comparison")),fontsize=20)
        ax.set_xlabel("Percentage of NA value",fontsize=18)
        ax.set_ylabel("MSE",fontsize=18)
        ax.set_yscale("log")
        ax.legend(legend,fontsize=12)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
    elif nb_mech==2:
        mse_MIDA_1=None
        mse_MIDA_2=None
        mse_MIWAE_1=None
        mse_MIWAE_2=None
        mse_MIWAE_std_1=None
        mse_MIWAE_std_2=None
        mse_RF_1=None
        mse_RF_2=None
        mse_MEAN_1=None
        mse_MEAN_2=None
        legend=[]
        mecha_1=mechanism_results[0]
        mecha_2=mechanism_results[1]
        mse_data_1=results[mecha_1]
        mse_data_2=results[mecha_2]
        for method in comp_meth:
            if method=="MIDA":
                mse_MIDA_1=mse_data_1.loc["MIDA"]
                mse_MIDA_2=mse_data_2.loc["MIDA"]
            elif method=="MIWAE":
                mse_MIWAE_1=mse_data_1.loc["MIWAE"]
                mse_MIWAE_2=mse_data_2.loc["MIWAE"]
            elif method=="MIWAE_standardization":
                mse_MIWAE_std_1=mse_data_1.loc["MIWAE_standardization"]
                mse_MIWAE_std_2=mse_data_2.loc["MIWAE_standardization"]
            elif method=="RF":
                mse_RF_1=mse_data_1.loc["RF"]
                mse_RF_2=mse_data_2.loc["RF"]
            elif method=="MEAN":
                mse_MEAN_1=mse_data_1.loc["MEAN"]
                mse_MEAN_2=mse_data_2.loc["MEAN"]
        f,(ax1,ax2)=plt.subplots(2,sharex=True,figsize=(7,8),constrained_layout=True)
        if mse_MIDA_1 is not None:
            ax1.plot(mse_MIDA_1,color="red")
            ax2.plot(mse_MIDA_2,color="red")
            legend+=["MIDA"]
        if mse_MIWAE_1 is not None:
            ax1.plot(mse_MIWAE_1,color="blue")
            ax2.plot(mse_MIWAE_2,color="blue")
            legend+=["MIWAE"]
        if mse_MIWAE_std_1 is not None:
            ax1.plot(mse_MIWAE_std_1,color="orchid")
            ax2.plot(mse_MIWAE_std_2,color="orchid")
            legend+=["MIWAE_standardization"]
        if mse_RF_1 is not None:
            ax1.plot(mse_RF_1,color="green")
            ax2.plot(mse_RF_2,color="green")
            legend+=["MissForest"]
        if mse_MEAN_1 is not None:
            ax1.plot(mse_MEAN_1,color="grey")
            ax2.plot(mse_MEAN_2,color="grey")
            legend+=["MEAN"]
        ax1.set_title(mecha_1,fontsize=20)
        ax2.set_title(mecha_2,fontsize=20)
        ax2.set_xlabel("Percentage of NA value",fontsize=18)
        ax1.set_ylabel("MSE",fontsize=18)
        ax2.set_ylabel("MSE",fontsize=18)
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.legend(legend,fontsize=12)
        ax2.legend(legend,fontsize=12)
        f.suptitle("Comparison of the different methods",fontsize=22)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
    elif nb_mech==3:
        mse_MIDA_1=None
        mse_MIDA_2=None
        mse_MIDA_3=None
        mse_MIWAE_1=None
        mse_MIWAE_2=None
        mse_MIWAE_3=None
        mse_MIWAE_std_1=None
        mse_MIWAE_std_2=None
        mse_MIWAE_std_3=None
        mse_RF_1=None
        mse_RF_2=None
        mse_RF_3=None
        mse_MEAN_1=None
        mse_MEAN_2=None
        mse_MEAN_3=None
        legend=[]
        mecha_1=mechanism_results[0]
        mecha_2=mechanism_results[1]
        mecha_3=mechanism_results[2]
        mse_data_1=results[mecha_1]
        mse_data_2=results[mecha_2]
        mse_data_3=results[mecha_3]
        for method in comp_meth:
            if method=="MIDA":
                mse_MIDA_1=mse_data_1.loc["MIDA"]
                mse_MIDA_2=mse_data_2.loc["MIDA"]
                mse_MIDA_3=mse_data_3.loc["MIDA"]
            elif method=="MIWAE":
                mse_MIWAE_1=mse_data_1.loc["MIWAE"]
                mse_MIWAE_2=mse_data_2.loc["MIWAE"]
                mse_MIWAE_3=mse_data_3.loc["MIWAE"]
            elif method=="MIWAE_standardization":
                mse_MIWAE_std_1=mse_data_1.loc["MIWAE_standardization"]
                mse_MIWAE_std_2=mse_data_2.loc["MIWAE_standardization"]
                mse_MIWAE_std_3=mse_data_3.loc["MIWAE_standardization"]
            elif method=="RF":
                mse_RF_1=mse_data_1.loc["RF"]
                mse_RF_2=mse_data_2.loc["RF"]
                mse_RF_3=mse_data_3.loc["RF"]
            elif method=="MEAN":
                mse_MEAN_1=mse_data_1.loc["MEAN"]
                mse_MEAN_2=mse_data_2.loc["MEAN"]
                mse_MEAN_3=mse_data_3.loc["MEAN"]
        f,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,figsize=(7,12),constrained_layout=True)
        if mse_MIDA_1 is not None:
            ax1.plot(mse_MIDA_1,color="red")
            ax2.plot(mse_MIDA_2,color="red")
            ax3.plot(mse_MIDA_3,color="red")
            legend+=["MIDA"]
        if mse_MIWAE_1 is not None:
            ax1.plot(mse_MIWAE_1,color="blue")
            ax2.plot(mse_MIWAE_2,color="blue")
            ax3.plot(mse_MIWAE_3,color="blue")
            legend+=["MIWAE"]
        if mse_MIWAE_std_1 is not None:
            ax1.plot(mse_MIWAE_std_1,color="orchid")
            ax2.plot(mse_MIWAE_std_2,color="orchid")
            ax3.plot(mse_MIWAE_std_3,color="orchid")
            legend+=["MIWAE_standardization"]
        if mse_RF_1 is not None:
            ax1.plot(mse_RF_1,color="green")
            ax2.plot(mse_RF_2,color="green")
            ax3.plot(mse_RF_3,color="green")
            legend+=["MissForest"]
        if mse_MEAN_1 is not None:
            ax1.plot(mse_MEAN_1,color="grey")
            ax2.plot(mse_MEAN_2,color="grey")
            ax3.plot(mse_MEAN_3,color="grey")
            legend+=["MEAN"]
        ax1.set_title(mecha_1,fontsize=20)
        ax2.set_title(mecha_2,fontsize=20)
        ax3.set_title(mecha_3,fontsize=20)
        ax3.set_xlabel("Percentage of NA value",fontsize=18)
        ax1.set_ylabel("MSE",fontsize=18)
        ax2.set_ylabel("MSE",fontsize=18)
        ax3.set_ylabel("MSE",fontsize=18)
        ax1.legend(legend,fontsize=12)
        ax2.legend(legend,fontsize=12)
        ax3.legend(legend,fontsize=12)
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax1.tick_params(axis="y",labelsize=16)
        ax2.tick_params(axis="y",labelsize=16)
        ax3.tick_params(axis="y",labelsize=16)
        ax1.tick_params(axis="x",labelsize=16)

        
        f.suptitle("Comparison of the different methods",fontsize=22)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
        
def box_plot(ax,data, edge_color,perc_l):
    bp = ax.boxplot(data, patch_artist=True,labels=perc_l)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor="white")       

    return bp

def boxplot_results(results,meth,prop_l,mech,save=False,name=None):
    nb_mecha=len(mech)
    if nb_mecha==1:
        
        key=mech[0].upper()
        mse_boxplot=results[key]
        perc_l=[" ".join((str(prop*100),"%")) for prop in prop_l]
        f,ax=plt.subplots(constrained_layout=True)
        legend=[]
        legend_color=[]

        if "MIDA" in meth:
            bp_MIDA=[mse_boxplot.loc["MIDA",perc] for perc in perc_l] 
            legend+=["MIDA"]   
            mida=box_plot(ax=ax,data=bp_MIDA,edge_color="#ff4937",perc_l=perc_l)
            legend_color+=[mida["boxes"][0]]
        if "MIWAE" in meth:
            bp_MIWAE=[mse_boxplot.loc["MIWAE",perc] for perc in perc_l] 
            legend+=["MIWAE"]   
            miwae=box_plot(ax=ax,data=bp_MIWAE,edge_color="#25189E",perc_l=perc_l)
            legend_color+=[miwae["boxes"][0]]
        if "MIWAE_standardization" in meth:
            bp_MIWAE_std=[mse_boxplot.loc["MIWAE_standardization",perc] for perc in perc_l] 
            legend+=["MIWAE_standardization"]   
            miwae_std=box_plot(ax=ax,data=bp_MIWAE_std,edge_color="orchid",perc_l=perc_l)
            legend_color+=[miwae_std["boxes"][0]]
        if "RF" in meth : 
            bp_RF=[mse_boxplot.loc["RF",perc] for perc in perc_l]
            legend+=["RF"]
            rf=box_plot(ax=ax,data=bp_RF,edge_color="#0e8a23",perc_l=perc_l)
            legend_color+=[rf["boxes"][0]]
        if "MEAN" in meth:
            bp_MEAN=[mse_boxplot.loc["MEAN",perc] for perc in perc_l] 
            legend+=["MEAN"]
            mean=box_plot(ax=ax,data=bp_MEAN,edge_color="#83808a",perc_l=perc_l)
            legend_color+=[mean["boxes"][0]]
        ax.set_xlabel("Percentage of NA values",fontsize=18)
        ax.set_ylabel("MSE",fontsize=18)
        ax.legend(legend_color,legend,fontsize=12)
        ax.set_title(key,fontsize=20)
        ax.tick_params(axis="x",labelsize=16)
        ax.tick_params(axis="y",labelsize=16)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
    elif nb_mecha==2:
        key_1=mech[0].upper()
        key_2=mech[1].upper()
        mse_boxplot_1=results[key_1]
        mse_boxplot_2=results[key_2]
        perc_l=[" ".join((str(prop*100),"%")) for prop in prop_l]
        f,(ax1,ax2)=plt.subplots(2,figsize=(8,15),sharex=True,constrained_layout=True)
        legend=[]
        legend_color=[]

        if "MIDA" in meth:
            bp_MIDA_1=[mse_boxplot_1.loc["MIDA",perc] for perc in perc_l] 
            bp_MIDA_2=[mse_boxplot_2.loc["MIDA",perc] for perc in perc_l] 
            legend+=["MIDA"]   
            mida_1=box_plot(ax=ax1,data=bp_MIDA_1,edge_color="#ff4937",perc_l=perc_l)
            mida_2=box_plot(ax=ax2,data=bp_MIDA_2,edge_color="#ff4937",perc_l=perc_l)
            legend_color+=[mida_1["boxes"][0]]
        if "MIWAE" in meth:
            bp_MIWAE_1=[mse_boxplot_1.loc["MIWAE",perc] for perc in perc_l] 
            bp_MIWAE_2=[mse_boxplot_2.loc["MIWAE",perc] for perc in perc_l] 
            legend+=["MIWAE"]   
            miwae_1=box_plot(ax=ax1,data=bp_MIWAE_1,edge_color="#25189E",perc_l=perc_l)
            miwae_2=box_plot(ax=ax2,data=bp_MIWAE_2,edge_color="#25189E",perc_l=perc_l)
            legend_color+=[miwae_1["boxes"][0]]
        if "MIWAE_standardization" in meth:
            bp_MIWAE_std_1=[mse_boxplot_1.loc["MIWAE_standardization",perc] for perc in perc_l] 
            bp_MIWAE_std_2=[mse_boxplot_2.loc["MIWAE_standardization",perc] for perc in perc_l] 
            legend+=["MIWAE_standardization"]   
            miwae_std_1=box_plot(ax=ax1,data=bp_MIWAE_std_1,edge_color="orchid",perc_l=perc_l)
            miwae_std_2=box_plot(ax=ax2,data=bp_MIWAE_std_2,edge_color="orchid",perc_l=perc_l)
            legend_color+=[miwae_std_1["boxes"][0]]
        if "RF" in meth : 
            bp_RF_1=[mse_boxplot_1.loc["RF",perc] for perc in perc_l]
            bp_RF_2=[mse_boxplot_2.loc["RF",perc] for perc in perc_l]
            legend+=["RF"]
            rf_1=box_plot(ax=ax1,data=bp_RF_1,edge_color="#0e8a23",perc_l=perc_l)
            rf_2=box_plot(ax=ax2,data=bp_RF_2,edge_color="#0e8a23",perc_l=perc_l)
            legend_color+=[rf_1["boxes"][0]]

        if "MEAN" in meth:
            bp_MEAN_1=[mse_boxplot_1.loc["MEAN",perc] for perc in perc_l] 
            bp_MEAN_2=[mse_boxplot_2.loc["MEAN",perc] for perc in perc_l] 
            legend+=["MEAN"]
            mean_1=box_plot(ax=ax1,data=bp_MEAN_1,edge_color="#83808a",perc_l=perc_l)
            mean_2=box_plot(ax=ax2,data=bp_MEAN_2,edge_color="#83808a",perc_l=perc_l)
            legend_color+=[mean_1["boxes"][0]]

        ax1.set_ylabel("MSE",fontsize=18)
        ax1.legend(legend_color,legend,fontsize=12)
        ax1.set_title(key_1,fontsize=20)
        ax2.set_xlabel("Percentage of NA values",fontsize=18)
        ax2.set_ylabel("MSE",fontsize=18)
        ax2.legend(legend_color,legend,fontsize=12)
        ax2.set_title(key_2,fontsize=18)
        ax1.tick_params(axis="y",labelsize=16)
        ax2.tick_params(axis="x",labelsize=16)
        ax2.tick_params(axis="y",labelsize=16)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
    elif nb_mecha==3:
        key_1=mech[0].upper()
        key_2=mech[1].upper()
        key_3=mech[2].upper()
        mse_boxplot_1=results[key_1]
        mse_boxplot_2=results[key_2]
        mse_boxplot_3=results[key_3]
        perc_l=[" ".join((str(prop*100),"%")) for prop in prop_l]
        f,(ax1,ax2,ax3)=plt.subplots(3,figsize=(8,16),sharex=True,constrained_layout=True)
        legend=[]
        legend_color=[]

        if "MIDA" in meth:
            bp_MIDA_1=[mse_boxplot_1.loc["MIDA",perc] for perc in perc_l] 
            bp_MIDA_2=[mse_boxplot_2.loc["MIDA",perc] for perc in perc_l]
            bp_MIDA_3=[mse_boxplot_3.loc["MIDA",perc] for perc in perc_l] 
            legend+=["MIDA"]   
            mida_1=box_plot(ax=ax1,data=bp_MIDA_1,edge_color="#ff4937",perc_l=perc_l)
            mida_2=box_plot(ax=ax2,data=bp_MIDA_2,edge_color="#ff4937",perc_l=perc_l)
            mida_3=box_plot(ax=ax3,data=bp_MIDA_3,edge_color="#ff4937",perc_l=perc_l)
            legend_color+=[mida_1["boxes"][0]]
        if "MIWAE" in meth:
            bp_MIWAE_1=[mse_boxplot_1.loc["MIWAE",perc] for perc in perc_l] 
            bp_MIWAE_2=[mse_boxplot_2.loc["MIWAE",perc] for perc in perc_l] 
            bp_MIWAE_3=[mse_boxplot_3.loc["MIWAE",perc] for perc in perc_l] 
            legend+=["MIWAE"]   
            miwae_1=box_plot(ax=ax1,data=bp_MIWAE_1,edge_color="#25189E",perc_l=perc_l)
            miwae_2=box_plot(ax=ax2,data=bp_MIWAE_2,edge_color="#25189E",perc_l=perc_l)
            miwae_3=box_plot(ax=ax3,data=bp_MIWAE_3,edge_color="#25189E",perc_l=perc_l)
            legend_color+=[miwae_1["boxes"][0]]
        if "MIWAE_standardization" in meth:
            bp_MIWAE_std_1=[mse_boxplot_1.loc["MIWAE_standardization",perc] for perc in perc_l] 
            bp_MIWAE_std_2=[mse_boxplot_2.loc["MIWAE_standardization",perc] for perc in perc_l] 
            bp_MIWAE_std_3=[mse_boxplot_3.loc["MIWAE_standardization",perc] for perc in perc_l] 
            legend+=["MIWAE_standardization"]   
            miwae_std_1=box_plot(ax=ax1,data=bp_MIWAE_std_1,edge_color="orchid",perc_l=perc_l)
            miwae_std_2=box_plot(ax=ax2,data=bp_MIWAE_std_2,edge_color="orchid",perc_l=perc_l)
            miwae_std_3=box_plot(ax=ax3,data=bp_MIWAE_std_3,edge_color="orchid",perc_l=perc_l)
            legend_color+=[miwae_std_1["boxes"][0]]
        if "RF" in meth : 
            bp_RF_1=[mse_boxplot_1.loc["RF",perc] for perc in perc_l]
            bp_RF_2=[mse_boxplot_2.loc["RF",perc] for perc in perc_l]
            bp_RF_3=[mse_boxplot_3.loc["RF",perc] for perc in perc_l]
            legend+=["RF"]
            rf_1=box_plot(ax=ax1,data=bp_RF_1,edge_color="#0e8a23",perc_l=perc_l)
            rf_2=box_plot(ax=ax2,data=bp_RF_2,edge_color="#0e8a23",perc_l=perc_l)
            rf_3=box_plot(ax=ax3,data=bp_RF_3,edge_color="#0e8a23",perc_l=perc_l)
            legend_color+=[rf_1["boxes"][0]]
        if "MEAN" in meth:
            bp_MEAN_1=[mse_boxplot_1.loc["MEAN",perc] for perc in perc_l] 
            bp_MEAN_2=[mse_boxplot_2.loc["MEAN",perc] for perc in perc_l] 
            bp_MEAN_3=[mse_boxplot_3.loc["MEAN",perc] for perc in perc_l] 
            legend+=["MEAN"]
            mean_1=box_plot(ax=ax1,data=bp_MEAN_1,edge_color="#83808a",perc_l=perc_l)
            mean_2=box_plot(ax=ax2,data=bp_MEAN_2,edge_color="#83808a",perc_l=perc_l)
            mean_3=box_plot(ax=ax3,data=bp_MEAN_3,edge_color="#83808a",perc_l=perc_l)
            legend_color+=[mean_1["boxes"][0]]

        ax1.set_ylabel("MSE",fontsize=18)
        ax1.legend(legend_color,legend,fontsize=12)
        ax1.set_title(key_1,fontsize=20)
        ax2.set_ylabel("MSE",fontsize=18)
        ax2.legend(legend_color,legend,fontsize=12)
        ax2.set_title(key_2,fontsize=20)
        ax3.set_ylabel("MSE",fontsize=18)
        ax3.legend(legend_color,legend,fontsize=12)
        ax3.set_title(key_3,fontsize=20)
        ax3.set_xlabel("Percentage of NA values",fontsize=18)
        ax1.tick_params(axis="y",labelsize=18)
        ax2.tick_params(axis="y",labelsize=18)
        ax3.tick_params(axis="y",labelsize=18)
        ax3.tick_params(axis="x",labelsize=18)
        if save :
            directory = os.getcwd()
            tmp_fig_path="\ ".join([directory,name])
            fig_path=tmp_fig_path.replace(" ","")
            plt.savefig(fig_path)
            
        


def plot_comparison_std(results,results_std,method,mech,save=False,name=None):
    nb_mech=len(mech)
    nb_meth=len(method)
    if nb_meth==1:
        if nb_mech==1:
            mse=results[mech[0]].loc[method[0]]
            mse_std=results_std[mech[0]].loc[method[0]]
            legend=[]
            f,ax=plt.subplots(constrained_layout=True)
            ax.plot(mse,color="red")
            legend+=["Standardization before introducing missing values"]
            ax.plot(mse_std,color="green")
            legend+=["Standardization after introducing missing values"]
            ax.set_ylabel("MSE",fontsize=18)
            ax.set_xlabel("Percentage of NA values",fontsize=18)
            ax.tick_params(axis="y",labelsize=16)
            ax.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the",method[0],"method between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax.set_title(" ".join([mech[0],"mechanism"]),fontsize=20)
            ax.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==2:
            mse_1=results[mech[0]].loc[method[0]]
            mse_std_1=results_std[mech[0]].loc[method[0]]
            mse_2=results[mech[1]].loc[method[0]]
            mse_std_2=results_std[mech[1]].loc[method[0]]
            legend=[]
            f,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(16,5),constrained_layout=True)
            ax1.plot(mse_1,color="red")
            ax2.plot(mse_2,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1,color="green")
            ax2.plot(mse_std_2,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax1.set_xlabel("Percentage of NA values",fontsize=18)
            ax2.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax1.tick_params(axis="x",labelsize=16)
            ax2.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the",method[0],"method between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([mech[1],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==3:
            mse_1=results[mech[0]].loc[method[0]]
            mse_std_1=results_std[mech[0]].loc[method[0]]
            mse_2=results[mech[1]].loc[method[0]]
            mse_std_2=results_std[mech[1]].loc[method[0]]
            mse_3=results[mech[2]].loc[method[0]]
            mse_std_3=results_std[mech[3]].loc[method[0]]
            legend=[]
            f,(ax1,ax2,ax3)=plt.subplots((1,3),sharey=True,constrained_layout=True)
            ax1.plot(mse_1,color="red")
            ax2.plot(mse_2,color="red")
            ax3.plot(mse_3,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1,color="green")
            ax2.plot(mse_std_2,color="green")
            ax3.plot(mse_std_3,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax1.set_xlabel("Percentage of NA values",fontsize=18)
            ax2.set_xlabel("Percentage of NA values",fontsize=18)
            ax3.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax1.tick_params(axis="x",labelsize=16)
            ax2.tick_params(axis="x",labelsize=16)
            ax3.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the",method[0],"method between \n a standardization done before or after introducing NA values"]),fontsiez=22)
            ax1.set_title(" ".join([mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([mech[2],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
    elif nb_meth==2:
        if nb_mech==1:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            legend=[]
            f,((ax1),(ax2))=plt.subplots(2,1,constrained_layout=True,figsize=(6,8),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_1_2,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_1_2,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax2.set_ylabel("MSE",fontsize=18)
            ax2.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax2.tick_params(axis="y",labelsize=16)
            ax2.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==2:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            legend=[]
            f,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,sharey=True,constrained_layout=True,figsize=(16,12),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_1_2,color="red")
            ax4.plot(mse_2_2,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_1_2,color="green")
            ax4.plot(mse_std_2_2,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax3.set_ylabel("MSE",fontsize=18)
            ax3.set_xlabel("Percentage of NA values",fontsize=18)
            ax4.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="x",labelsize=16)
            ax4.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==3:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_3_1=results[mech[2]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_std_3_1=results_std[mech[2]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_3_2=results[mech[2]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            mse_std_3_2=results_std[mech[2]].loc[method[1]]
            legend=[]
            f,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,sharey=True,constrained_layout=True,sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_3_1,color="red")
            ax4.plot(mse_1_2,color="red")
            ax5.plot(mse_2_2,color="red")
            ax6.plot(mse_3_2,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_3_1,color="green")
            ax4.plot(mse_std_1_2,color="green")
            ax5.plot(mse_std_2_2,color="green")
            ax6.plot(mse_std_3_2,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax4.set_ylabel("MSE",fontsize=18)
            ax4.set_xlabel("Percentage of NA values",fontsize=18)
            ax5.set_xlabel("Percentage of NA values",fontsize=18)
            ax6.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax4.tick_params(axis="y",labelsize=16)
            ax5.tick_params(axis="x",labelsize=16)
            ax4.tick_params(axis="x",labelsize=16)
            ax6.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[0],"method,",mech[2],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax5.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax6.set_title(" ".join([method[1],"method,",mech[2],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            ax5.legend(legend,fontsize=12)
            ax6.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
    elif nb_meth==3:
        if nb_mech==1:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            legend=[]
            f,((ax1),(ax2),(ax3))=plt.subplots(3,1,constrained_layout=True,figsize=(6,8),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_1_2,color="red")
            ax3.plot(mse_1_3,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_1_2,color="green")
            ax3.plot(mse_std_1_3,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax2.set_ylabel("MSE",fontsize=18)
            ax3.set_ylabel("MSE",fontsize=18)
            ax3.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax2.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[2],"method,",mech[0],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==2:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_2_3=results[mech[1]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            mse_std_2_3=results_std[mech[1]].loc[method[2]]
            legend=[]
            f,((ax1,ax2),(ax3,ax4),(ax5,ax6))=plt.subplots(3,2,sharey=True,constrained_layout=True,figsize=(16,14),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_1_2,color="red")
            ax4.plot(mse_2_2,color="red")
            ax5.plot(mse_1_3,color="red")
            ax6.plot(mse_2_3,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_1_2,color="green")
            ax4.plot(mse_std_2_2,color="green")
            ax5.plot(mse_std_1_3,color="green")
            ax6.plot(mse_std_2_3,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax3.set_ylabel("MSE",fontsize=18)
            ax5.set_ylabel("MSE",fontsize=18)
            ax5.set_xlabel("Percentage of NA values",fontsize=18)
            ax6.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="y",labelsize=16)
            ax5.tick_params(axis="y",labelsize=16)
            ax5.tick_params(axis="x",labelsize=16)
            ax6.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax5.set_title(" ".join([method[2],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax6.set_title(" ".join([method[2],"method,",mech[1],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            ax5.legend(legend,fontsize=12)
            ax6.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==3:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_3_1=results[mech[2]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_std_3_1=results_std[mech[2]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_3_2=results[mech[2]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            mse_std_3_2=results_std[mech[2]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_2_3=results[mech[1]].loc[method[2]]
            mse_3_3=results[mech[2]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            mse_std_2_3=results_std[mech[1]].loc[method[2]]
            mse_std_3_3=results_std[mech[2]].loc[method[2]]
            legend=[]
            f,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9))=plt.subplots(3,3,sharey=True,constrained_layout=True,sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_3_1,color="red")
            ax4.plot(mse_1_2,color="red")
            ax5.plot(mse_2_2,color="red")
            ax6.plot(mse_3_2,color="red")
            ax7.plot(mse_1_3,color="red")
            ax8.plot(mse_2_3,color="red")
            ax9.plot(mse_3_3,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_3_1,color="green")
            ax4.plot(mse_std_1_2,color="green")
            ax5.plot(mse_std_2_2,color="green")
            ax6.plot(mse_std_3_2,color="green")
            ax7.plot(mse_std_1_3,color="green")
            ax8.plot(mse_std_2_3,color="green")
            ax9.plot(mse_std_3_3,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax4.set_ylabel("MSE",fontsize=18)
            ax7.set_ylabel("MSE",fontsize=18)
            ax7.set_xlabel("Percentage of NA values",fontsize=18)
            ax8.set_xlabel("Percentage of NA values",fontsize=18)
            ax9.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax4.tick_params(axis="y",labelsize=16)
            ax7.tick_params(axis="y",labelsize=16)
            ax7.tick_params(axis="x",labelsize=16)
            ax8.tick_params(axis="x",labelsize=16)
            ax9.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[0],"method,",mech[2],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax5.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax6.set_title(" ".join([method[1],"method,",mech[2],"mechanism"]),fontsize=20)
            ax7.set_title(" ".join([method[2],"method,",mech[0],"mechanism"]),fontsize=20)
            ax8.set_title(" ".join([method[2],"method,",mech[1],"mechanism"]),fontsize=20)
            ax9.set_title(" ".join([method[2],"method,",mech[2],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            ax5.legend(legend,fontsize=12)
            ax6.legend(legend,fontsize=12)
            ax7.legend(legend,fontsize=12)
            ax8.legend(legend,fontsize=12)
            ax9.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
    elif nb_meth==4:
        if nb_mech==1:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            mse_1_4=results[mech[0]].loc[method[3]]
            mse_std_1_4=results_std[mech[0]].loc[method[3]]
            legend=[]
            f,((ax1),(ax2),(ax3),(ax4))=plt.subplots(4,1,constrained_layout=True,figsize=(8,16),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_1_2,color="red")
            ax3.plot(mse_1_3,color="red")
            ax4.plot(mse_1_4,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_1_2,color="green")
            ax3.plot(mse_std_1_3,color="green")
            ax4.plot(mse_std_1_4,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax2.set_ylabel("MSE",fontsize=18)
            ax3.set_ylabel("MSE",fontsize=18)
            ax4.set_ylabel("MSE",fontsize=18)
            ax4.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax2.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="y",labelsize=16)
            ax4.tick_params(axis="y",labelsize=16)
            ax4.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[2],"method",mech[0],"mechanism"]),fontsize=20)
            ax4.set_title(" ".join([method[3],"method",mech[0],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==2:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_2_3=results[mech[1]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            mse_std_2_3=results_std[mech[1]].loc[method[2]]
            mse_1_4=results[mech[0]].loc[method[3]]
            mse_2_4=results[mech[1]].loc[method[3]]
            mse_std_1_4=results_std[mech[0]].loc[method[3]]
            mse_std_2_4=results_std[mech[1]].loc[method[3]]
            legend=[]
            f,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8))=plt.subplots(4,2,sharey=True,constrained_layout=True,figsize=(16,18),sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_1_2,color="red")
            ax4.plot(mse_2_2,color="red")
            ax5.plot(mse_1_3,color="red")
            ax6.plot(mse_2_3,color="red")
            ax7.plot(mse_1_4,color="red")
            ax8.plot(mse_2_4,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_1_2,color="green")
            ax4.plot(mse_std_2_2,color="green")
            ax5.plot(mse_std_1_3,color="green")
            ax6.plot(mse_std_2_3,color="green")
            ax7.plot(mse_std_1_4,color="green")
            ax8.plot(mse_std_2_4,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax3.set_ylabel("MSE",fontsize=18)
            ax5.set_ylabel("MSE",fontsize=18)
            ax7.set_ylabel("MSE",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax3.tick_params(axis="y",labelsize=16)
            ax5.tick_params(axis="y",labelsize=16)
            ax7.tick_params(axis="y",labelsize=16)
            ax7.set_xlabel("Percentage of NA values",fontsize=18)
            ax8.set_xlabel("Percentage of NA values",fontsize=18)
            ax7.tick_params(axis="x",labelsize=16)
            ax8.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax5.set_title(" ".join([method[2],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax6.set_title(" ".join([method[2],"method,",mech[1],"mechanism"]),fontsize=20)
            ax7.set_title(" ".join([method[3],"method,",mech[0],"mechanism"]),fontsize=20)  
            ax8.set_title(" ".join([method[3],"method,",mech[1],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            ax5.legend(legend,fontsize=12)
            ax6.legend(legend,fontsize=12)
            ax7.legend(legend,fontsize=12)
            ax8.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        elif nb_mech==3:
            mse_1_1=results[mech[0]].loc[method[0]]
            mse_2_1=results[mech[1]].loc[method[0]]
            mse_3_1=results[mech[2]].loc[method[0]]
            mse_std_1_1=results_std[mech[0]].loc[method[0]]
            mse_std_2_1=results_std[mech[1]].loc[method[0]]
            mse_std_3_1=results_std[mech[2]].loc[method[0]]
            mse_1_2=results[mech[0]].loc[method[1]]
            mse_2_2=results[mech[1]].loc[method[1]]
            mse_3_2=results[mech[2]].loc[method[1]]
            mse_std_1_2=results_std[mech[0]].loc[method[1]]
            mse_std_2_2=results_std[mech[1]].loc[method[1]]
            mse_std_3_2=results_std[mech[2]].loc[method[1]]
            mse_1_3=results[mech[0]].loc[method[2]]
            mse_2_3=results[mech[1]].loc[method[2]]
            mse_3_3=results[mech[2]].loc[method[2]]
            mse_std_1_3=results_std[mech[0]].loc[method[2]]
            mse_std_2_3=results_std[mech[1]].loc[method[2]]
            mse_std_3_3=results_std[mech[2]].loc[method[2]]
            mse_1_4=results[mech[0]].loc[method[3]]
            mse_2_4=results[mech[1]].loc[method[3]]
            mse_3_4=results[mech[2]].loc[method[3]]
            mse_std_1_4=results_std[mech[0]].loc[method[3]]
            mse_std_2_4=results_std[mech[1]].loc[method[3]]
            mse_std_3_4=results_std[mech[2]].loc[method[3]]
            legend=[]
            f,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12))=plt.subplots(4,3,sharey=True,constrained_layout=True,sharex=True)
            ax1.plot(mse_1_1,color="red")
            ax2.plot(mse_2_1,color="red")
            ax3.plot(mse_3_1,color="red")
            ax4.plot(mse_1_2,color="red")
            ax5.plot(mse_2_2,color="red")
            ax6.plot(mse_3_2,color="red")
            ax7.plot(mse_1_3,color="red")
            ax8.plot(mse_2_3,color="red")
            ax9.plot(mse_3_3,color="red")
            ax10.plot(mse_1_4,color="red")
            ax11.plot(mse_2_4,color="red")
            ax12.plot(mse_3_4,color="red")
            legend+=["Standardization before introducing missing values"]
            ax1.plot(mse_std_1_1,color="green")
            ax2.plot(mse_std_2_1,color="green")
            ax3.plot(mse_std_3_1,color="green")
            ax4.plot(mse_std_1_2,color="green")
            ax5.plot(mse_std_2_2,color="green")
            ax6.plot(mse_std_3_2,color="green")
            ax7.plot(mse_std_1_3,color="green")
            ax8.plot(mse_std_2_3,color="green")
            ax9.plot(mse_std_3_3,color="green")
            ax10.plot(mse_std_1_4,color="green")
            ax11.plot(mse_std_2_4,color="green")
            ax12.plot(mse_std_3_4,color="green")
            legend+=["Standardization after introducing missing values"]
            ax1.set_ylabel("MSE",fontsize=18)
            ax4.set_ylabel("MSE",fontsize=18)
            ax7.set_ylabel("MSE",fontsize=18)
            ax10.set_ylabel("MSE",fontsize=18)
            ax10.set_xlabel("Percentage of NA values",fontsize=18)
            ax11.set_xlabel("Percentage of NA values",fontsize=18)
            ax12.set_xlabel("Percentage of NA values",fontsize=18)
            ax1.tick_params(axis="y",labelsize=16)
            ax4.tick_params(axis="y",labelsize=16)
            ax7.tick_params(axis="y",labelsize=16)
            ax10.tick_params(axis="y",labelsize=16)
            ax10.tick_params(axis="x",labelsize=16)
            ax11.tick_params(axis="x",labelsize=16)
            ax12.tick_params(axis="x",labelsize=16)
            f.suptitle(" ".join(["Comparison for the different methods between \n a standardization done before or after introducing NA values"]),fontsize=22)
            ax1.set_title(" ".join([method[0],"method,",mech[0],"mechanism"]),fontsize=20)
            ax2.set_title(" ".join([method[0],"method,",mech[1],"mechanism"]),fontsize=20)
            ax3.set_title(" ".join([method[0],"method,",mech[2],"mechanism"]),fontsize=20)  
            ax4.set_title(" ".join([method[1],"method,",mech[0],"mechanism"]),fontsize=20)
            ax5.set_title(" ".join([method[1],"method,",mech[1],"mechanism"]),fontsize=20)
            ax6.set_title(" ".join([method[1],"method,",mech[2],"mechanism"]),fontsize=20)
            ax7.set_title(" ".join([method[2],"method,",mech[0],"mechanism"]),fontsize=20)
            ax8.set_title(" ".join([method[2],"method,",mech[1],"mechanism"]),fontsize=20)
            ax9.set_title(" ".join([method[2],"method,",mech[2],"mechanism"]),fontsize=20)
            ax10.set_title(" ".join([method[3],"method,",mech[0],"mechanism"]),fontsize=20)
            ax11.set_title(" ".join([method[3],"method,",mech[1],"mechanism"]),fontsize=20)
            ax12.set_title(" ".join([method[3],"method,",mech[2],"mechanism"]),fontsize=20)
            ax1.legend(legend,fontsize=12)
            ax2.legend(legend,fontsize=12)
            ax3.legend(legend,fontsize=12)
            ax4.legend(legend,fontsize=12)
            ax5.legend(legend,fontsize=12)
            ax6.legend(legend,fontsize=12)
            ax7.legend(legend,fontsize=12)
            ax8.legend(legend,fontsize=12)
            ax9.legend(legend,fontsize=12)
            ax10.legend(legend,fontsize=12)
            ax11.legend(legend,fontsize=12)
            ax12.legend(legend,fontsize=12)
            if save:
                directory = os.getcwd()
                tmp_fig_path="\ ".join([directory,name])
                fig_path=tmp_fig_path.replace(" ","")
                plt.savefig(fig_path)
            plt.show()
        
        

        
        
    
        

        
        
    


            