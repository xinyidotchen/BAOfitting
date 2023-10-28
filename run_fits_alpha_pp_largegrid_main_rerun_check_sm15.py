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

import fits_functions_alpha_pp_largegrid1 as fits_fun
import fits_minimize_functions as mini

import sys
import configparser

#N=np.int(sys.argv[1]) # grid size
#L=np.int(sys.argv[2]) # box size
#category=np.str(sys.argv[3])
#subcategory=np.str(sys.argv[4])
#redshift=np.float(sys.argv[5])
#space=np.str(sys.argv[6])
#sample=np.str(sys.argv[7])
#smooth_scale=np.float(sys.argv[8])
#mass_lim_low=np.float(sys.argv[9])
#mass_lim_up=np.float(sys.argv[10])
#Sig_perp=np.float(sys.argv[11])
#Sig_para=np.float(sys.argv[12])
#f=np.float(sys.argv[13])
#bias=np.float(sys.argv[14])
config_file=np.str(sys.argv[1])

config=configparser.ConfigParser()
config.read(config_file)

cov_mat_file=config['input']['cov_file']
N=np.int(config['input']['N'])
L=np.int(config['input']['L'])
category=config['input']['category']
subcategory=config['input']['subcategory']
redshift=np.float(config['input']['redshift'])
space=config['input']['space']
sample=config['input']['sample']
smooth_scale=np.float(config['input']['smooth_scale'])
mass_lim_low=np.float(config['input']['mass_lim_low'])
mass_lim_up=np.float(config['input']['mass_lim_up'])
Sig_perp=np.float(config['input']['Sig_perp'])
Sig_para=np.float(config['input']['Sig_para'])
f=np.float(config['input']['f'])
bias=np.float(config['input']['bias'])
largegrid_num=np.str(config['input']['bias'])



k_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=0)
Plin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=1)
Pnw_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=3)
Olin_Nikhil=np.genfromtxt("quijote_smooth.dat",usecols=-1)

growth_factor0=0.7892460926468668
growth_factor1=0.4782343934210456

if redshift==float(1):
	Pnw_Nikhil=Pnw_Nikhil/(growth_factor0/growth_factor1)**2


Olin_model_Nikhil=interpolate.interp1d(k_Nikhil,Olin_Nikhil,fill_value='extrapolate')
Pnw_model=interpolate.interp1d(np.log(k_Nikhil),np.log(Pnw_Nikhil),fill_value='extrapolate')



k=np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/Snapshots/fiducial_HR/0/grid512/z1.0/bf_rec/multi_pk_bf_rec_spacez_numPl3.txt",usecols=0)

mask=(k<0.4) & (k>0.03) 
k_cut=k[mask]

IC_alpha=np.genfromtxt("IC_alpha_0.03_0.4.txt",usecols=0)

#if category=='Snapshots':
#	cov_mat=np.load("analytic_cov_snap_z%i_%sspace_%s_k_0.03_0.4_dk_0.01.dat.npy"%(redshift,space,sample,smooth_scale))
#if category=='Halos':
#	cov_mat=np.load("analytic_cov_halo%.2f-%.2f_z%i_%sspace_%s_k_0.03_0.4_dk_0.01.dat.npy"%(mass_lim_low,mass_lim_up,redshift,space,sample,smooth_scale))
cov_mat=np.load(cov_mat_file)


invcov_full=np.linalg.inv(cov_mat/2)



if category=='Snapshots':
	if sample =='standard':
		data_P0_numPl3=[]
		for i in range(0,100):
			data_P0_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/1024cubex8randoms/multi_pk_%s_space%s_ani1.0_numPl3.txt"%(category,i,redshift,sample,smooth_scale,sample,space),usecols=1,unpack=True))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(0,100):
			data_P0_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/1024cubex8randoms/multi_pk_%s_space%s_ani1.0_numPl5.txt"%(category,i,redshift,sample,smooth_scale,sample,space),usecols=1,unpack=True))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]


		data_P2_numPl3=[]
		for i in range(0,100):
			data_P2_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/1024cubex8randoms/multi_pk_%s_space%s_ani1.0_numPl3.txt"%(category,i,redshift,sample,smooth_scale,sample,space),usecols=2,unpack=True))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(0,100):
			data_P2_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/1024cubex8randoms/multi_pk_%s_space%s_ani1.0_numPl5.txt"%(category,i,redshift,sample,smooth_scale,sample,space),usecols=2,unpack=True))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]

	if sample =='bf_rec':
		data_P0_numPl3=[]
		for i in range(0,100):
			data_P0_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/multi_pk_%s_space%s_numPl3.txt"%(category,i,redshift,sample,sample,space),usecols=1,unpack=True))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(0,100):
			data_P0_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/multi_pk_%s_space%s_numPl5.txt"%(category,i,redshift,sample,sample,space),usecols=1,unpack=True))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]


		data_P2_numPl3=[]
		for i in range(0,100):
			data_P2_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/multi_pk_%s_space%s_numPl3.txt"%(category,i,redshift,sample,sample,space),usecols=2,unpack=True))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(0,100):
			data_P2_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/multi_pk_%s_space%s_numPl5.txt"%(category,i,redshift,sample,sample,space),usecols=2,unpack=True))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]



	if sample =='hada':
		data_P0_numPl3=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl3.txt"%(category,i,redshift,sample,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)

			data_P0_numPl3.append(np.genfromtxt(latest_file,usecols=3))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl5.txt"%(category,i,redshift,sample,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P0_numPl5.append(np.genfromtxt(latest_file,usecols=3))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]



		data_P2_numPl3=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl3.txt"%(category,i,redshift,sample,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P2_numPl3.append(np.genfromtxt(latest_file,usecols=4))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl5.txt"%(category,i,redshift,sample,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P2_numPl5.append(np.genfromtxt(latest_file,usecols=4))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]





if category=='Halos':
	if sample=='standard':
		data_P0_numPl3=[]
		for i in range(0,100):
			data_P0_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_%s_space%s_ani1.0_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale,sample,space),usecols=1,unpack=True))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(0,100):
			data_P0_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_%s_space%s_ani1.0_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale,sample,space),usecols=1,unpack=True))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]


		data_P2_numPl3=[]
		for i in range(0,100):
			data_P2_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_%s_space%s_ani1.0_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale,sample,space),usecols=2,unpack=True))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(0,100):
			data_P2_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_%s_space%s_ani1.0_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale,sample,space),usecols=2,unpack=True))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]
	
	if sample=='bf_rec':
		data_P0_numPl3=[]
		for i in range(0,100):
			data_P0_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/multi_pk_%s_space%s_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,sample,space),usecols=1,unpack=True))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(0,100):
			data_P0_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/multi_pk_%s_space%s_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,sample,space),usecols=1,unpack=True))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]


		data_P2_numPl3=[]
		for i in range(0,100):
			data_P2_numPl3.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/multi_pk_%s_space%s_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,sample,space),usecols=2,unpack=True))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(0,100):
			data_P2_numPl5.append(np.genfromtxt("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/multi_pk_%s_space%s_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,sample,space),usecols=2,unpack=True))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]




	if sample =='hada':
		data_P0_numPl3=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)

			data_P0_numPl3.append(np.genfromtxt(latest_file,usecols=3))
		data_P0_numPl3=np.array(data_P0_numPl3)

		data_P0_numPl5=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P0_numPl5.append(np.genfromtxt(latest_file,usecols=3))
		data_P0_numPl5=np.array(data_P0_numPl5)

		data_P0_comb=data_P0_numPl5
		data_P0_comb[:,:10]=data_P0_numPl3[:,:10]



		data_P2_numPl3=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl3.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P2_numPl3.append(np.genfromtxt(latest_file,usecols=4))
		data_P2_numPl3=np.array(data_P2_numPl3)

		data_P2_numPl5=[]
		for i in range(100):
			files=glob.glob("/home/xc298/project/reconstruction_project/output/Quijote/%s/fiducial_HR/%i/grid512/z%.1f/%s/%.2f-%.2f/sm%i/multi_pk_hada_spacez_ani1.0_iter*_numPl5.txt"%(category,i,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale))
			latest_file= max(files,key=os.path.getmtime)
			data_P2_numPl5.append(np.genfromtxt(latest_file,usecols=4))
		data_P2_numPl5=np.array(data_P2_numPl5)

		data_P2_comb=data_P2_numPl5
		data_P2_comb[:,:10]=data_P2_numPl3[:,:10]








batch_num=50
data_P0_comb_split=np.array_split(data_P0_comb,batch_num)
data_P0_comb_split_mean=np.mean(data_P0_comb_split,axis=1)[:,mask]

data_P2_comb_split=np.array_split(data_P2_comb,batch_num)
data_P2_comb_split_mean=np.mean(data_P2_comb_split,axis=1)[:,mask]



# minimize
d=np.append(data_P0_comb_split_mean,data_P2_comb_split_mean,axis=1).T

alpha_range=np.array([1])
epsilon_range=np.array([0])
ae_grid_a,ae_grid_e,ae_grid_k_cut=np.meshgrid(alpha_range,epsilon_range,k_cut,indexing='ij')


chi2=np.zeros(50)
Sig_s_vals=np.zeros(50)
R_vals=np.zeros(50)
initial=[2,15]


for i in range(50):
    
	minimize_res=minimize(mini.Likelihood_function,initial,args=(i,ae_grid_k_cut,ae_grid_a,ae_grid_e,Sig_perp,Sig_para,f,bias,redshift,d,invcov_full),method='Nelder-Mead')
	chi2[i]=minimize_res.fun
	R_vals[i]=minimize_res.x[1]
	Sig_s_vals[i]=minimize_res.x[0]

savearr_Sig_s_Sig_R=np.array([Sig_s_vals,R_vals,chi2]).T

if category=='Snapshots':
	filename='/home/xc298/project/reconstruction_project/output/Quijote_BAOfits/'\
		+'%s/z%.1f/%s/sm%.1f/'%(category,redshift,sample,smooth_scale)+'alpha_pp/cov_extrashotnoise/fits_Sig_s_Sig_R_largegrid_num%s.txt'%(largegrid_num)

if category=='Halos':
	filename='/home/xc298/project/reconstruction_project/output/Quijote_BAOfits/'\
		+'%s/z%.1f/%s/%.2f-%.2f/sm%.1f/'%(category,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale)+'alpha_pp/cov_extrashotnoise/fits_Sig_s_Sig_R_largegrid_num%s.txt'%(largegrid_num)	

os.makedirs(os.path.dirname(filename), exist_ok=True)
np.savetxt(filename,savearr_Sig_s_Sig_R,fmt=['%.12f','%.12f','%.12f'],\
	header='%12s\t%12s\t%12s'%('Sig_s','Sig_R','chi2'),delimiter='\t')





# run fits

# a-e grid
#alpha_perp_range=np.arange(0.95,1.05,0.001)
#alpha_para_range=np.arange(0.9,1.1,0.001)

#ae_grid_a,ae_grid_e,ae_grid_k_cut=np.meshgrid(alpha_perp_range,alpha_para_range,k_cut,indexing='ij')

#k arrays
k_arr=k_cut
one_arr=np.ones(len(k_cut))
kinv1_arr=1./k_cut
kinv2_arr=1./k_cut**2
kinv3_arr=1./k_cut**3
k2_arr=k_cut**2


Ptemp0_arr=[]
Ptemp2_arr=[]
for i in range(50):
	Ptemp0_arr_ele,Ptemp2_arr_ele=fits_fun.run_Ptemp(Sig_s_vals[i],Sig_perp,Sig_para,f,bias,R_vals[i],redshift)
	Ptemp0_arr.append(Ptemp0_arr_ele)
	Ptemp2_arr.append(Ptemp2_arr_ele)
    
Ptemp0_arr=np.array(Ptemp0_arr)
Ptemp2_arr=np.array(Ptemp2_arr)


sol_arr=fits_fun.run_wo_prior(Ptemp0_arr,Ptemp2_arr,d,invcov_full)

print("d shape=",np.shape(d))
print("sol_arr shape=",np.shape(sol_arr))
print("Ptemp0_arr shape=",np.shape(Ptemp0_arr))
print("Ptemp2_arr shape=",np.shape(Ptemp2_arr))
chi2_arr,min_chi2,min_chi2_pdof,min_chi2_alpha_perp,min_chi2_alpha_para=fits_fun.run_wo_prior_chi2(d,sol_arr,Ptemp0_arr,Ptemp2_arr,invcov_full)

if category=='Snapshots':
	filename='/home/xc298/project/reconstruction_project/output/Quijote_BAOfits/'\
		+'%s/z%.1f/%s/sm%.1f/'%(category,redshift,sample,smooth_scale)+'alpha_pp/cov_extrashotnoise/fits_chi2_largegrid_num%s.dat'%(largegrid_num)

if category=='Halos':
	filename='/home/xc298/project/reconstruction_project/output/Quijote_BAOfits/'\
		+'%s/z%.1f/%s/%.2f-%.2f/sm%.1f/'%(category,redshift,sample,mass_lim_low,mass_lim_up,smooth_scale)+'alpha_pp/cov_extrashotnoise/fits_chi2_largegrid_num%s.dat'%(largegrid_num)

os.makedirs(os.path.dirname(filename), exist_ok=True)
np.save(filename,chi2_arr)

