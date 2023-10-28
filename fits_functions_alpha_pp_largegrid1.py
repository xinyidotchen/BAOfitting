
import numpy as np
import numpy.random as npr
import timeit
import scipy.special
from scipy import interpolate

from scipy.integrate import quad
from scipy.integrate import nquad

import math
import os,time,datetime
import re
import glob
from scipy.optimize import minimize
from scipy.optimize import fmin


## a-e grid
#alpha_perp_range=np.arange(0.95,1.05,0.0005)
#alpha_para_range=np.arange(-0.15,0.151,0.001)
#
#ae_grid_a,ae_grid_e,ae_grid_k_cut=np.meshgrid(alpha_perp_range,alpha_para_range,k_cut,indexing='ij')
#
##k arrays
#k_arr=k_cut
#one_arr=np.ones(len(k_cut))
#kinv1_arr=1./k_cut
#kinv2_arr=1./k_cut**2
#kinv3_arr=1./k_cut**3
#k2_arr=k_cut**2


#functions

k_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=0)
Plin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=1)
Pnw_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=3)
Olin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=-1)



Olin_model_Nikhil=interpolate.interp1d(k_Nikhil,Olin_Nikhil,fill_value='extrapolate')

k=np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/Snapshots/fiducial_HR/0/grid512/z1.0/bf_rec/multi_pk_bf_rec_spacez_numPl3.txt",usecols=0)

mask=(k<0.4) & (k>0.03)
k_cut=k[mask]

# a-e grid
#alpha_perp_range=np.arange(0.95,1.05,0.001)
#alpha_para_range=np.arange(0.9,1.1,0.001)
#alpha_perp_range=np.arange(0.9,1.1,0.0005)
#alpha_para_range=np.arange(0.85,1.15,0.0005)
alpha_perp_range=np.arange(0.9,1.1,0.0025)
alpha_para_range=np.arange(0.85,1.15,0.0025)
#alpha_perp_range=np.arange(0.9,1.1,0.001)
#alpha_para_range=np.arange(0.85,1.15,0.001)
#alpha_perp_range=np.arange(0.85,1.15,0.0025)
#alpha_para_range=np.arange(0.8,1.2,0.0025)

ae_grid_a,ae_grid_e,ae_grid_k_cut=np.meshgrid(alpha_perp_range,alpha_para_range,k_cut,indexing='ij')

#k arrays
k_arr=k_cut
one_arr=np.ones(len(k_cut))
kinv1_arr=1./k_cut
kinv2_arr=1./k_cut**2
kinv3_arr=1./k_cut**3
k2_arr=k_cut**2


def F(input_k,input_mu,Sig_s):
    return 1./(1.+input_k**2*input_mu**2*Sig_s**2/2)**2
def Pdw(input_k,input_mu,Sig_perp,Sig_para,f,redshift):
#     Sig_para=(1+f)*Sig_perp
    growth_factor0=0.7892460926468668
    growth_factor1=0.4782343934210456
    k_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=0)
    Plin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=1)
    Pnw_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=3)
    Olin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=-1)
    if redshift==float(1):
        Pnw_Nikhil=Pnw_Nikhil/(growth_factor0/growth_factor1)**2
    Pnw_model=interpolate.interp1d(np.log(k_Nikhil),np.log(Pnw_Nikhil),fill_value='extrapolate')

    k_para2=input_k**2*input_mu**2
    k_perp2=(1-input_mu**2)*input_k**2

    return np.exp(Pnw_model(np.log(input_k)))*(1+(Olin_model_Nikhil(input_k)-1)*np.exp(-1./2.*(input_k**2*input_mu**2*Sig_para**2+input_k**2*(1-input_mu**2)*Sig_perp**2)))
def Ptemp_kmu(input_k,input_mu,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift):
    alpha=alpha_para**(1./3.)*alpha_perp**(2./3.)
    epsilon=(alpha_para/alpha_perp)**(1./3.)-1
    alpha_ratio=(1+epsilon)**3
    mu_prime=input_mu/alpha_ratio/np.sqrt(1+input_mu**2*(1/alpha_ratio**2-1))
#    alpha_perp=alpha/(1+epsilon)
    k_prime=input_k/alpha_perp*np.sqrt(1+input_mu**2*(1/alpha_ratio**2-1))
#    alpha_para=alpha*(1+epsilon)**2
    prefac=1/alpha_perp**2/alpha_para
    R=1-np.exp(-(k_prime*smoothing)**2/2)

    beta=f/b

    return prefac*(1.+beta*mu_prime**2.*R)**2.*F(k_prime,mu_prime,Sig_s)*Pdw(k_prime,mu_prime,Sig_perp,Sig_para,f,redshift)

def Ptemp0_k_integrand(input_mu,input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift): # monopole
    return Ptemp_kmu(input_k,input_mu,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift)*scipy.special.legendre(0)(input_mu)
def Ptemp0_k(input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift):
    mu,weight=np.polynomial.legendre.leggauss(deg=20)
    return (2*0+1.)/2.*np.sum(weight*Ptemp0_k_integrand(mu,input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift),axis=3)

def Ptemp2_k_integrand(input_mu,input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift): # quadrupole
    return Ptemp_kmu(input_k,input_mu,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift)*scipy.special.legendre(2)(input_mu)

def Ptemp2_k(input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift):
    mu,weight=np.polynomial.legendre.leggauss(deg=20)
    return (2*2+1.)/2.*np.sum(weight*Ptemp2_k_integrand(mu,input_k,alpha_perp,alpha_para,Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift),axis=3)


def run_Ptemp(Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift):
    Ptemp0=np.array(Ptemp0_k(ae_grid_k_cut[:,:,:,np.newaxis],ae_grid_a[:,:,:,np.newaxis],ae_grid_e[:,:,:,np.newaxis],Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift))
    Ptemp2=np.array(Ptemp2_k(ae_grid_k_cut[:,:,:,np.newaxis],ae_grid_a[:,:,:,np.newaxis],ae_grid_e[:,:,:,np.newaxis],Sig_s,Sig_perp,Sig_para,f,b,smoothing,redshift))
    return Ptemp0,Ptemp2



def wo_prior_grid(Ptemp0,Ptemp2):
    

    Ptemp0220_ele=np.append(Ptemp0,Ptemp2,axis=-1)

    k_arr_02=np.append(k_arr,np.zeros(len(k_cut)))
    k_arr_02=np.broadcast_to(k_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(k_arr_02)))


    k_arr_20=np.append(np.zeros(len(k_cut)),k_arr)
    k_arr_20=np.broadcast_to(k_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(k_arr_20)))

    one_arr_02=np.append(one_arr,np.zeros(len(k_cut)))
    one_arr_02=np.broadcast_to(one_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(one_arr_02)))

    one_arr_20=np.append(np.zeros(len(k_cut)),one_arr)
    one_arr_20=np.broadcast_to(one_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(one_arr_20)))

    kinv1_arr_02=np.append(kinv1_arr,np.zeros(len(k_cut)))
    kinv1_arr_02=np.broadcast_to(kinv1_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(kinv1_arr_02)))

    kinv1_arr_20=np.append(np.zeros(len(k_cut)),kinv1_arr)
    kinv1_arr_20=np.broadcast_to(kinv1_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(kinv1_arr_20)))

    kinv2_arr_02=np.append(kinv2_arr,np.zeros(len(k_cut)))
    kinv2_arr_02=np.broadcast_to(kinv2_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(kinv2_arr_02)))

    kinv2_arr_20=np.append(np.zeros(len(k_cut)),kinv2_arr)
    kinv2_arr_20=np.broadcast_to(kinv2_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(kinv2_arr_20)))

    kinv3_arr_02=np.append(kinv3_arr,np.zeros(len(k_cut)))
    kinv3_arr_02=np.broadcast_to(kinv3_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(kinv3_arr_02)))

    kinv3_arr_20=np.append(np.zeros(len(k_cut)),kinv3_arr)
    kinv3_arr_20=np.broadcast_to(kinv3_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(kinv3_arr_20)))


    k2_arr_02=np.append(k2_arr,np.zeros(len(k_cut)))
    k2_arr_02=np.broadcast_to(k2_arr_02,(len(alpha_perp_range),len(alpha_para_range),len(k2_arr_02)))

    k2_arr_20=np.append(np.zeros(len(k_cut)),k2_arr)
    k2_arr_20=np.broadcast_to(k2_arr_20,(len(alpha_perp_range),len(alpha_para_range),len(k2_arr_20)))

    return Ptemp0220_ele,k_arr_02,k_arr_20,one_arr_02,one_arr_20,kinv1_arr_02,kinv1_arr_20,kinv2_arr_02,kinv2_arr_20,kinv3_arr_02,kinv3_arr_20,k2_arr_02,k2_arr_20






def run_wo_prior(Ptemp0,Ptemp2,d,invcov_full):
    L=np.linalg.cholesky(invcov_full)
    LT=L.T.conj()
    
    LTd=LT.dot(d)

    
    sol_arr=[]

    for dd in range(len(LTd[0])):

        Ptemp0220_ele,k_arr_02,k_arr_20,one_arr_02,one_arr_20,kinv1_arr_02,kinv1_arr_20,kinv2_arr_02,kinv2_arr_20,kinv3_arr_02,kinv3_arr_20,k2_arr_02,k2_arr_20=wo_prior_grid(Ptemp0[dd,:,:,:],Ptemp2[dd,:,:,:])
    
        mat=np.array([Ptemp0220_ele,k_arr_02,k_arr_20,one_arr_02,one_arr_20,kinv1_arr_02,kinv1_arr_20,kinv2_arr_02,kinv2_arr_20,kinv3_arr_02,kinv3_arr_20,k2_arr_02,k2_arr_20])
        mat=np.transpose(mat,(1,2,3,0))
        mat=LT.dot(mat)
        mat=np.transpose(mat,(0,3,1,2))
        
        sol=[np.linalg.lstsq(mat[:,:,i,j],LTd[:,dd],rcond=None)[0] for i in range(len(alpha_perp_range)) for j in range(len(alpha_para_range))]        

        sol=np.reshape(sol,(len(alpha_perp_range),len(alpha_para_range),13))

        sol_arr.append(sol)
    sol_arr=np.array(sol_arr)
    
    return sol_arr

def run_wo_prior_fit4Sig_avgallsims(Ptemp0,Ptemp2,d,invcov_full):
    L=np.linalg.cholesky(invcov_full)
    LT=L.T.conj()
    
    LTd=LT.dot(d)

    
    sol_arr=[]

    Ptemp0220_ele,k_arr_02,k_arr_20,one_arr_02,one_arr_20,kinv1_arr_02,kinv1_arr_20,kinv2_arr_02,kinv2_arr_20,kinv3_arr_02,kinv3_arr_20,k2_arr_02,k2_arr_20=wo_prior_grid(Ptemp0[0,:,:,:],Ptemp2[0,:,:,:])
    mat=np.array([Ptemp0220_ele,k_arr_02,k_arr_20,one_arr_02,one_arr_20,kinv1_arr_02,kinv1_arr_20,kinv2_arr_02,kinv2_arr_20,kinv3_arr_02,kinv3_arr_20,k2_arr_02,k2_arr_20])
    mat=np.transpose(mat,(1,2,3,0))
    mat=LT.dot(mat)
    mat=np.transpose(mat,(0,3,1,2))

    for dd in range(len(LTd[0])):

        
        sol=[np.linalg.lstsq(mat[:,:,i,j],LTd[:,dd],rcond=None)[0] for i in range(len(alpha_perp_range)) for j in range(len(alpha_para_range))]        

        sol=np.reshape(sol,(len(alpha_perp_range),len(alpha_para_range),13))

        sol_arr.append(sol)
    sol_arr=np.array(sol_arr)
    
    return sol_arr


# chi2
def run_wo_prior_chi2(d,sol_arr,Ptemp0_arr,Ptemp2_arr,invcov_full):
    chi2_arr=np.zeros((len(d[0]),len(alpha_perp_range),len(alpha_para_range)))
    print("chi2_arr shape=",np.shape(chi2_arr))
    for dd in range(len(d[0])):
        for i in range(len(alpha_perp_range)):
            for j in range(len(alpha_para_range)):
                fit0=sol_arr[dd,i,j,0]*Ptemp0_arr[dd,i,j,:]+sol_arr[dd,i,j,1]*k_cut+sol_arr[dd,i,j,3]+sol_arr[dd,i,j,5]/k_cut+sol_arr[dd,i,j,7]/k_cut**2+sol_arr[dd,i,j,9]/k_cut**3+sol_arr[dd,i,j,11]*k_cut**2
                fit2=sol_arr[dd,i,j,0]*Ptemp2_arr[dd,i,j,:]+sol_arr[dd,i,j,2]*k_cut+sol_arr[dd,i,j,4]+sol_arr[dd,i,j,6]/k_cut+sol_arr[dd,i,j,8]/k_cut**2+sol_arr[dd,i,j,10]/k_cut**3+sol_arr[dd,i,j,12]*k_cut**2
                fit=np.append(fit0,fit2)
                chi2_arr[dd,i,j]=np.dot(np.dot((d[:,dd]-fit),(invcov_full)),(d[:,dd]-fit))
                
    min_chi2=np.min(np.min(chi2_arr,axis=-1),axis=-1)
    min_chi2_pdof=min_chi2/(2*len(k_cut)-15)

    unravel_index=[(np.unravel_index(chi2_arr[i,:,:].argmin(),chi2_arr[i,:,:].shape)) for i in range(len(d[0]))]
    unravel_index=np.array(unravel_index)

    min_chi2_alpha_perp=alpha_perp_range[unravel_index[:,0]]
    min_chi2_alpha_para=np.around(alpha_para_range[unravel_index[:,1]],3)

    return chi2_arr,min_chi2,min_chi2_pdof,min_chi2_alpha_perp,min_chi2_alpha_para

def run_wo_prior_chi2_fit4Sig_avgallsims(d,sol_arr,Ptemp0_arr,Ptemp2_arr,invcov_full):
    chi2_arr=np.zeros((len(d[0]),len(alpha_perp_range),len(alpha_para_range)))
    print("chi2_arr shape=",np.shape(chi2_arr))
    for dd in range(len(d[0])):
        for i in range(len(alpha_perp_range)):
            for j in range(len(alpha_para_range)):
                fit0=sol_arr[dd,i,j,0]*Ptemp0_arr[0,i,j,:]+sol_arr[dd,i,j,1]*k_cut+sol_arr[dd,i,j,3]+sol_arr[dd,i,j,5]/k_cut+sol_arr[dd,i,j,7]/k_cut**2+sol_arr[dd,i,j,9]/k_cut**3+sol_arr[dd,i,j,11]*k_cut**2
                fit2=sol_arr[dd,i,j,0]*Ptemp2_arr[0,i,j,:]+sol_arr[dd,i,j,2]*k_cut+sol_arr[dd,i,j,4]+sol_arr[dd,i,j,6]/k_cut+sol_arr[dd,i,j,8]/k_cut**2+sol_arr[dd,i,j,10]/k_cut**3+sol_arr[dd,i,j,12]*k_cut**2
                fit=np.append(fit0,fit2)
                chi2_arr[dd,i,j]=np.dot(np.dot((d[:,dd]-fit),(invcov_full)),(d[:,dd]-fit))
                
    min_chi2=np.min(np.min(chi2_arr,axis=-1),axis=-1)
    min_chi2_pdof=min_chi2/(2*len(k_cut)-15)

    unravel_index=[(np.unravel_index(chi2_arr[i,:,:].argmin(),chi2_arr[i,:,:].shape)) for i in range(len(d[0]))]
    unravel_index=np.array(unravel_index)

    min_chi2_alpha_perp=alpha_perp_range[unravel_index[:,0]]
    min_chi2_alpha_para=np.around(alpha_para_range[unravel_index[:,1]],3)

    return chi2_arr,min_chi2,min_chi2_pdof,min_chi2_alpha_perp,min_chi2_alpha_para
