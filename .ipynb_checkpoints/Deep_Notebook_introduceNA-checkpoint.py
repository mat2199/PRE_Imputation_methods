import pandas as pd
import numpy as np

## Comments

## Home-made model
#if (modSimu == 3){
 # tab <- matrix(NA,nind,10)
 # X <- runif(nind,-2,2)
 # tab[,1] <- X
 # tab[,2] = X^2
 # tab[,3] = (abs(X)-1)*(X< 0)+ sin(5*X)*(X>0)
 # tab[,4] = cos(X*10)
 # tab[,5] = exp(X)
 # tab[,6] = 2*(X< -1)+ 1*(X>-1 & X<0) + 2*(X<1 & X>0) + 1* (X<2 & X>1)
 # tab[,7] = 1*(X< -1)+ 1*(X>-1 & X<0) + 2*(X<1 & X>0) + 1* (X<2 & X>1)
 # tab[,8] = 1*(X< -0.5) - 2.5*X*(X>-0.5 & X<0) + 1*(X<1.5 & X>0) + X/2* (X<2 & X>1.5)
 # tab[,9] = (X+2)*(X< -1)+ X^2*(X>-1 & X<0) + (exp(X)-2)*(X<1 & X>0) + cos(X)*(X<2 & X>1)
 # tab[,10] = -(X^2-3)*(X< -1)+ -exp(X)*(X>-1 & X<0) + sin(10*X)*(X<1 & X>0) + (exp(X)-4)*(X<2 & X>1)
 # tab <- scale(tab,center=TRUE, scale=TRUE)
 # tab <- tab + matrix(rnorm(ncol(tab)*nrow(tab),0,bruit),ncol=ncol(tab))
 # tab <- scale(tab,center=TRUE, scale=TRUE)
#}

### Imputation of the missing values using the mean
def imput_mean(data,mask):
    col = data.shape[1]
    mask_moy = np.invert(mask)
    for m in range(0,col,1):
        column = data.iloc[:,m]
        moy = np.mean(column[mask[:,m]])
        data.iloc[mask_moy[:,m],m] = moy
    return data


### Function to introduce missing values
def missing_method(raw_data, mechanism='mcar', method='uniform', t = 0.2) :
    
    data = raw_data.copy()
    rows, cols = data.shape
    
    # missingness threshold
    t = t
    
    if mechanism == 'mcar' :
    
        if method == 'uniform' :
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
            mask = (v<=t)
            data[mask] = 0

        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True

            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
            mask = (v<=t)*c
            data[mask] = 0

        else :
            print("Error : There are no such method")
            raise
    
    elif mechanism == 'mar' :
        data = np.array(data)
        if method == 'uniform' :
            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)

            # calculate ther median m1, m2
            m1, m2 = np.median(data[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
#            m1 = data[:,sample_cols[0]] <= m1
#            m2 = data[:,sample_cols[1]] >= m2
#            m = (m1*m2)[:, np.newaxis]
            
            mask = (v<=0)  # put only 0's
            mask[data[:,sample_cols[0]] <= m1,sample_cols[0]] = 1
            mask[data[:,sample_cols[1]] <= m2,sample_cols[1]] = 1

#            mask = m*(v<=t)
            data[mask] = 0
            data = pd.DataFrame(data)
            mask = np.array(mask)

        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True

            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)

            # calculate ther median m1, m2
            m1, m2 = np.median(data[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = data[:,sample_cols[0]] <= m1
            m2 = data[:,sample_cols[1]] >= m2
            m = (m1*m2)[:, np.newaxis]

            mask = m*(v<=t)*c
            data[mask] = 0
            data = pd.DataFrame(data)
            mask = np.array(mask)

        else :
            print("Error : There is no such method")
            raise
    
    else :
        print("Error : There is no such mechanism")
        raise
        
    return data, mask