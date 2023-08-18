import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def f1(u,t,rho=0.5):
            return (1/2*np.pi*np.sqrt(1-np.power(rho,2)))*np.exp(-(np.power(u,2)-2*rho*u*t+np.power(t,2))/(2*(1-np.power(rho,2))))

def gfun(s1=0.5):
    return s1

def hfun(x):
    return np.inf


def f2(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-np.power(x,2)/2)

rho_array=np.arange(-0.99,0.99,0.1)


def prob(s1,,rho_array=rho_array,f2=f2,gfun=gfun,hfun=hfun):
    prob=[]
    for rho in rho_array:
        def f1(u,t,rho=rho):
            return (1/2*np.pi*np.sqrt(1-np.power(rho,2)))*np.exp(-(np.power(u,2)-2*rho*u*t+np.power(t,2))/(2*(1-np.power(rho,2))))
        num=scipy.integrate.dblquad(f1, s1,np.inf , gfun(s1=s1), hfun)[0]
        den=scipy.integrate.quad(f2, s1, np.inf)[0]
        prob.append(num/den)

    return prob

prob_test=prob(s1=0.5)

plt.plot(prob_test)



    
    