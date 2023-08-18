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
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])


def prior(d,device):
    p_z = td.Independent(td.Normal(loc=torch.zeros(d).to(device),scale=torch.ones(d).to(device)),1)
    return (p_z)



def miwae_loss(encoder,decoder,d,p,K,p_z,iota_x,mask):
  batch_size = iota_x.shape[0]
  out_encoder = encoder(iota_x)
  q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
  
  zgivenx = q_zgivenxobs.rsample([K])
  zgivenx_flat = zgivenx.reshape([K*batch_size,d])
  
  out_decoder = decoder(zgivenx_flat)
  all_means_obs_model = out_decoder[..., :p]
  all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
  all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
  
  data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
  tiledmask = torch.Tensor.repeat(mask,[K,1])
  
  all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
  all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])
  
  logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
  logpz = p_z.log_prob(zgivenx)
  logq = q_zgivenxobs.log_prob(zgivenx)
  
  neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
  
  return neg_bound

## A function that will return the encoder and the decoder of our MIWAE procedure

def build_encoder_decoder(p,h=128,d=1):
  decoder = nn.Sequential(
      torch.nn.Linear(d, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
  )

  encoder = nn.Sequential(
     torch.nn.Linear(p, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
  )
  return encoder,decoder

# A function that returns the optimiser
def build_optimizer(encoder,decoder):

  optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-3)
  return(optimizer)

def miwae_impute(encoder,decoder,d,p,p_z,iota_x,mask,L,device):
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)
  
    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])
  
    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
    all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3
  
    data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).to(device)
    tiledmask = torch.Tensor.repeat(mask,[L,1]).to(device)
  
    all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])
  
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
  
    xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    xms = xgivenz.sample().reshape([L,batch_size,p])
    xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
  

  
    return xm



def weights_init(layer):
  if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)

# Ajout moyenne initiale et écart type initial 

def train_MIWAE(encoder,decoder,optimizer,d,p_z,miss_data,raw_data,device,n_epochs=2002,bs=64,K=20,verbose=False):
    n,p=raw_data.shape
    miwae_loss_train=np.array([])
    mse_train=np.array([])
    mse_train2=np.array([])
    mask1=np.isfinite(miss_data)
    mask=np.copy(mask1)
    xhat0=np.copy(miss_data)
    xhat0[np.isnan(miss_data)]=0
    xhat = np.copy(xhat0)# This will be out imputed data matrix
    xfull=np.copy(raw_data)
    
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    for ep in range(1,n_epochs):
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        batches_data = np.array_split(xhat0[perm,], n/bs)
        batches_mask = np.array_split(mask[perm,], n/bs)
        for it in range(len(batches_data)):
            optimizer.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            b_data = torch.from_numpy(batches_data[it]).float().to(device)
            b_mask = torch.from_numpy(batches_mask[it]).float().to(device)
            loss = miwae_loss(encoder=encoder,decoder=decoder,d=d,p=p,K=K,p_z=p_z,iota_x = b_data,mask = b_mask)
            loss.backward()
            optimizer.step()
        if ep % 100 == 1:
            if verbose:
                
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(encoder=encoder,decoder=decoder,d=d,p=p,K=K,p_z=p_z,iota_x = torch.from_numpy(xhat0).float().to(device),mask =torch.from_numpy(mask).float().to(device)).cpu().data.numpy())) # Gradient step      
    
        ### Now we do the imputation
    
            xhat[~mask] = miwae_impute(encoder=encoder,decoder=decoder,d=d,p=p,p_z=p_z,device=device,iota_x = torch.from_numpy(xhat0).float().to(device),mask = torch.from_numpy(mask).float().to(device),L=10).cpu().data.numpy()[~mask]
            err = np.array([mse(xhat,xfull,mask)])
            mse_train = np.append(mse_train,err,axis=0)
            # Calcul moyenne et écart type et mettre à jour 
            # Déstandardise et restandardiser
            
            if verbose :
                
                print('Imputation MSE  %g' %err)
                print('-----')
    return (mse_train)


def miwae_multiple_impute(data,encoder,decoder,d,p,p_z,device,M=10,L=10):
    miwae_imputation=[]
    xhat0=np.copy(data)
    xhat0[np.isnan(data)]=0
    mask=np.copy(np.isfinite(data))
    
    for i in range (M):
        imp=np.copy(xhat0)
        imp[~mask]=miwae_impute(encoder=encoder,decoder=decoder,d=d,p=p,p_z=p_z,iota_x=torch.from_numpy(xhat0).float().to(device),device=device,L=L,mask = torch.from_numpy(mask).float().to(device)).cpu().data.numpy()[~mask]
        miwae_imputation.append(pd.DataFrame(imp))
    return miwae_imputation
        
def avg_mse(list_imp,xtrue,mask):
    list_mse=[]
    for i in range(len(list_imp)):
        list_mse.append(mse(xhat=list_imp[i],xtrue=xtrue,mask=mask))
    return mean(list_mse)


def train_MIWAE_standardization(encoder,decoder,optimizer,d,p_z,miss_data,raw_data,mean_0,std_0,device,thr=100,int=20,n_epochs=2001,bs=64,K=20,verbose=False):
    n,p=raw_data.shape
    miwae_loss_train=np.array([])
    mse_train=np.array([])
    mse_train2=np.array([])
    mask1=np.isfinite(miss_data)
    mask=np.copy(mask1)
    xhat0=np.copy(miss_data)
    xhat0[np.isnan(miss_data)]=0
    xhat = np.copy(xhat0)# This will be out imputed data matrix
    xfull=np.copy(raw_data)
    
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    old_mean=mean_0
    old_std=std_0

    for ep in range(1,n_epochs):
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        batches_data = np.array_split(xhat0[perm,], n/bs)
        batches_mask = np.array_split(mask[perm,], n/bs)
        for it in range(len(batches_data)):
            optimizer.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            b_data = torch.from_numpy(batches_data[it]).float().to(device)
            b_mask = torch.from_numpy(batches_mask[it]).float().to(device)
            loss = miwae_loss(encoder=encoder,decoder=decoder,d=d,p=p,K=K,p_z=p_z,iota_x = b_data,mask = b_mask)
            loss.backward()
            optimizer.step()
        if ep % int == 1 and ep>thr:
            if verbose:
                
                print('Epoch %g' %ep)
                print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(encoder=encoder,decoder=decoder,d=d,p=p,K=K,p_z=p_z,iota_x = torch.from_numpy(xhat0).float().to(device),mask =torch.from_numpy(mask).float().to(device)).cpu().data.numpy())) # Gradient step      
    
        ### Now we do the imputation
    
            xhat[~mask] = miwae_impute(encoder=encoder,decoder=decoder,d=d,p=p,p_z=p_z,device=device,iota_x = torch.from_numpy(xhat0).float().to(device),mask = torch.from_numpy(mask).float().to(device),L=10).cpu().data.numpy()[~mask]
            err = np.array([mse(xhat,xfull,mask)])
            mse_train = np.append(mse_train,err,axis=0)
            
            destandardized_data=np.multiply(pd.DataFrame(xhat),old_std)+old_mean
            new_mean=np.mean(destandardized_data,0)
            new_std=np.std(destandardized_data,0)
            new_standardized_data=(destandardized_data-new_mean)/new_std
            old_mean=new_mean
            old_std=new_std
            xhat0=new_standardized_data.to_numpy()
            xhat=xhat0.copy()
            
            if verbose :
                
                print('Imputation MSE  %g' %err)
                print('-----')
    return (mse_train)

def plot_mean_MIWAE(raw_data,mean,ep,mech,prop_NA,save=False,name=None):
    m=np.around(mean,4)
    nb_val=len(mean[0][0])
    legend=[]
    nb_mech=len(mech)
    nb_ep=len(ep[0])
    if nb_mech==1:
        f,ax=plt.subplots(1,constrained_layout=True,figsize=(10,6))
        for i in range(nb_val):
            val=[m[0][j][i] for j in range (nb_ep)]-np.mean(raw_data,0)[i]
            ax.plot(ep[0],val)
            legend+=["_".join(["var",str(i)])]
        ax.legend(legend,fontsize=12)
        ax.set_xlabel("Training epochs",fontsize=16)
        ax.axhline(y=0,color="black")
        ax.set_ylabel("Bias on the mean",fontsize=16)
        ax.set_title(" ".join([mech[0].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
        f.suptitle("Evolution of the bias on the mean \n on our imputed data during the model training",fontsize=20)
        ax.tick_params(axis="x",labelsize=16)
        ax.tick_params(axis="y",labelsize=16)
        plt.show()
    if nb_mech==2:
        f,(ax1,ax2)=plt.subplots(1,2,constrained_layout=True,figsize=(14,6),sharey=True)
        for k in range (nb_mech):
            if k==0:
                for i in range(nb_val):
                    val=[m[k][j][i] for j in range (nb_ep)]-np.mean(raw_data,0)[i]
                    ax1.plot(ep[0],val)
                    legend+=["_".join(["var",str(i)])]
                ax1.legend(legend,fontsize=12)
                ax1.set_xlabel("Training epochs",fontsize=16)
                ax1.axhline(y=0,color="black")
                ax1.set_ylabel("Bias on the mean",fontsize=16)
                ax1.set_title(" ".join([mech[0].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
                f.suptitle("Evolution of the bias on the mean \n on our imputed data during the model training",fontsize=20)
                ax1.tick_params(axis="x",labelsize=16)
                ax1.tick_params(axis="y",labelsize=16)
            if k==1:
                for i in range(nb_val):
                    val=[m[k][j][i] for j in range (nb_ep)]-np.mean(raw_data,0)[i]
                    ax2.plot(ep[0],val)
                ax2.legend(legend,fontsize=12)
                ax2.set_xlabel("Training epochs",fontsize=16)
                ax2.axhline(y=0,color="black")
                ax2.set_title(" ".join([mech[1].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
                ax2.tick_params(axis="x",labelsize=16)
                ax2.tick_params(axis="y",labelsize=16)
        plt.show()

def plot_std_MIWAE(raw_data,std,ep,mech,prop_NA,save=False,name=None):
    m=np.around(std,4)
    nb_val=len(std[0][0])
    legend=[]
    nb_mech=len(mech)
    nb_ep=len(ep[0])
    if nb_mech==1:
        f,ax=plt.subplots(1,constrained_layout=True,figsize=(10,6))
        for i in range(nb_val):
            val=[m[0][j][i] for j in range (nb_ep)]-np.std(raw_data,0)[i]
            ax.plot(ep[0],val)
            legend+=["_".join(["var",str(i)])]
        ax.legend(legend,fontsize=12)
        ax.set_xlabel("Training epochs",fontsize=16)
        ax.axhline(y=0,color="black")
        ax.set_ylabel("Bias on the standard deviation",fontsize=16)
        ax.set_title(" ".join([mech[0].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
        f.suptitle("Evolution of the bias on the standard deviation \n on our imputed data during the model training",fontsize=20)
        ax.tick_params(axis="x",labelsize=16)
        ax.tick_params(axis="y",labelsize=16)
        plt.show()
    if nb_mech==2:
        f,(ax1,ax2)=plt.subplots(1,2,constrained_layout=True,figsize=(14,6),sharey=True)
        for k in range (nb_mech):
            if k==0:
                for i in range(nb_val):
                    val=[m[k][j][i] for j in range (nb_ep)]-np.std(raw_data,0)[i]
                    ax1.plot(ep[0],val)
                    legend+=["_".join(["var",str(i)])]
                ax1.legend(legend,fontsize=12)
                ax1.set_xlabel("Training epochs",fontsize=16)
                ax1.axhline(y=0,color="black")
                ax1.set_ylabel("Bias on the standard deviation",fontsize=16)
                ax1.set_title(" ".join([mech[0].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
                f.suptitle("Evolution of the bias on the standard deviation \n on our imputed data during the model training",fontsize=20)
                ax1.tick_params(axis="x",labelsize=16)
                ax1.tick_params(axis="y",labelsize=16)
            if k==1:
                for i in range(nb_val):
                    val=[m[k][j][i] for j in range (nb_ep)]-np.std(raw_data,0)[i]
                    ax2.plot(ep[0],val)
                ax2.legend(legend,fontsize=12)
                ax2.set_xlabel("Training epochs",fontsize=16)
                ax2.axhline(y=0,color="black")
                ax2.set_title(" ".join([mech[1].upper(),"mechanism with",str(prop_NA*100),"% of NA values"]),fontsize=18)
                ax2.tick_params(axis="x",labelsize=16)
                ax2.tick_params(axis="y",labelsize=16)
        plt.show()