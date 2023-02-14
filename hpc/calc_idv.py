
import argparse
import json
import numpy as np
import math
import matplotlib as mpl
mpl.use('agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing
import scipy

import pynbody.filt as flt
import pynbody.analysis.luminosity as lum

import os, time

from scipy.optimize import leastsq

from functions import *



plt.rc('text',usetex=False)



### options parsing

parser = argparse.ArgumentParser(description='calc and plot')
parser.add_argument('-d', '--datbase', type=str, help='database')
parser.add_argument('-i', '--idx', type=int, help='index')
parser.add_argument('-g', '--galname', type=str, help='galname')
args = parser.parse_args()


pynbody.config['number_of_threads'] = 1
os.environ['OMP_NUM_THREADS']='1'


base = '/data/database/nihao/'
directories = dict()
directories['nihao_classic'] = '/data/mb6605/nihao/nihao_classic'
directories['nihao_bh'] = base+'nihao_agn/nihao_bh'
directories['nihao_ell_bh'] = base+'nihao_agn/nihao_ell_bh'
directories['nihao_ell_wobh'] = base+'nihao_agn/nihao_ell_wobh'
directories['nihao_nadine'] = '/data/mb6605/nihao/nihao_nadine'


###
##### Set parameters and definitions
###

nts = 64

sfr_type = 'sfr_200Myr_mass'
sfr_time = 200.e6     #in years

ts_Gyr = 0.215769412

starf = flt.HighPass('tform', '0 Gyr')
bhf = flt.LowPass('tform', '0 Gyr')
f_agets = flt.LowPass('age','215.769412 Myr')





###
##### Load all data
###

keys = ['mbh_c','sfr_ts','nsfr_ts','star_age_median','star_age_mean','star_age_std','star_metals_median','star_metals_mean','star_metals_std','mstar','nstar','time','a_scale','mvir','u-r','time_alt','u-r_alt','r_shm3D','rh_sfr_ts','rh_nsfr_ts','rh_star_age_median','rh_star_age_mean','rh_star_age_std','rh_star_metals_median','rh_star_metals_mean','rh_star_metals_std','rh_mstar','rh_nstar','rh_u-r','rh_u-r_alt','kappa_co']

keys += ['sin_u-r','sin_u_alt','sin_r_alt','sin_u_new','sin_r_new','sin_u-r_alt','bin_u-r','bin_u_alt','bin_r_alt','bin_u_new','bin_r_new','bin_u-r_alt']
keys += ['rh_u-r','rh_u_alt','rh_r_alt','rh_u_new','rh_r_new','rh_u-r_alt','rh_sin_u-r','rh_sin_u_alt','rh_sin_r_alt','rh_sin_u_new','rh_sin_r_new','rh_sin_u-r_alt','rh_bin_u-r','rh_bin_u_alt','rh_bin_r_alt','rh_bin_u_new','rh_bin_r_new','rh_bin_u-r_alt']

def calc_lums(tempdct, stars, i, prf='', lfile='cmdlum.npz'):
    tempdct[prf+'u-r'] = float(halo_mag(stars, band='u', lumfile=lfile)-halo_mag(stars, band='r', lumfile=lfile))
    tempdct[prf+'u_alt'] = [float(halo_mag(stars, band='u', addage=1e9*k*ts_Gyr, lumfile=lfile)) for k in range(i+1)]
    tempdct[prf+'r_alt'] = [float(halo_mag(stars, band='r', addage=1e9*k*ts_Gyr, lumfile=lfile)) for k in range(i+1)]
    tempdct[prf+'u_new'] = [float(halo_mag(stars[f_agets], band='u', addage=1e9*k*ts_Gyr, lumfile=lfile)) for k in range(i+1)]
    tempdct[prf+'r_new'] = [float(halo_mag(stars[f_agets], band='r', addage=1e9*k*ts_Gyr, lumfile=lfile)) for k in range(i+1)]
    tempdct[prf+'u-r_alt'] = list(np.array(tempdct[prf+'u_alt'])-np.array(tempdct[prf+'r_alt']))


def calc_strs(tempdct, stars, i, prf=''):
    tempdct[prf+'sfr_ts'] = float(np.sum(stars[f_agets]['mass'].in_units('Msol')) / (ts_Gyr*1e9))
    tempdct[prf+'nsfr_ts'] = len(stars[f_agets])
    tempdct[prf+'star_age_median'] = float(np.median(stars['age'].in_units('Gyr')))
    tempdct[prf+'star_age_mean'] = float(np.mean(stars['age'].in_units('Gyr')))
    tempdct[prf+'star_age_std'] = float(np.std(stars['age'].in_units('Gyr')))
    tempdct[prf+'star_metals_median'] = float(np.median(stars['metals']))
    tempdct[prf+'star_metals_mean'] = float(np.mean(stars['metals']))
    tempdct[prf+'star_metals_std'] = float(np.std(stars['metals']))
    tempdct[prf+'mstar'] = float(np.sum(stars['mass'].in_units('Msol')))
    tempdct[prf+'nstar'] = len(stars)
    ## Calculate u and r band luminosities
    calc_lums(tempdct, stars, i, prf=prf+'', lfile='cmdlum.npz')
    calc_lums(tempdct, stars, i, prf=prf+'bin_', lfile='bpass_bin_imf135_300.npz')
    calc_lums(tempdct, stars, i, prf=prf+'sin_', lfile='bpass_sin_imf135_300.npz')
    ##
    tempdct[prf+'mass_new'] = float(np.sum(stars[f_agets]['mass'].in_units('Msol')))
    if (tempdct[prf+'mass_new']>0.0):
        tempdct[prf+'age_new'] = float(np.sum(stars[f_agets]['age'].in_units('Gyr')*stars[f_agets]['mass'].in_units('Msol')))/tempdct['mass_new']
    else:
        tempdct[prf+'age_new'] = 0.0


def dosnap(prms):
    tempdct = dict()

    for key in keys:
        tempdct[key] = np.nan

    datbase = prms[0]
    i = prms[1]
    galname = prms[2]

    print('start', datbase, galname, i)

    filename = directories[datbase]+'/'+galname+'/haloids.json'
    if os.path.exists(filename):
        f2 = open(filename, 'r') 
        haloids = json.load(f2)
        f2.close()
    else: return tempdct

    N = len(np.where(np.array(haloids['haloids'][0])>=0)[0])
    if (N==0): return tempdct

    s=pynbody.load(directories[datbase]+'/'+galname+'/'+galname[:8]+'.'+str(haloids['timestep'][i]).zfill(5))
    
    s.physical_units()
    h=s.halos()

    if (len(h)==0): return tempdct

    hid = haloids['haloids'][0][i]+1
    for j in range(1,len(h)+1):
        if (h[j].properties['halo_id'] == hid): break

    center_avd(h[j])

    rvir = h[j]['r'].in_units('kpc').max()
    r20f = flt.Sphere(0.2*rvir)

    stars = h[j].s
    if (len(stars)==0): return tempdct
    stars = stars[starf]
    if (len(stars)==0): return tempdct
    stars = stars[r20f]
    if (len(stars)==0): return tempdct

    r_shm3D = h_smooth(stars,N=int(0.5*len(stars))).in_units('kpc')
    
    try:
       bhs = h[j].s[bhf][r20f]
       i_bh = bhs['mass'].argmax()
       tempdct['mbh_c'] = bhs['mass'][i_bh]
    except:
        pass

    tempdct['n_steps'] = N
    tempdct['r_shm3D'] = float(r_shm3D)
    tempdct['time'] = float(s.properties['time'].in_units('Gyr'))
    tempdct['a_scale'] = float(s.properties['a'])
    tempdct['mvir'] = float(np.sum(h[j]['mass'].in_units('Msol')))
    tempdct['time_alt'] = [tempdct['time']+k*ts_Gyr for k in range(i+1)]

    calc_strs(tempdct, stars, i, prf='')
    calc_strs(tempdct, stars[flt.Sphere(r_shm3D)], i, prf='rh_')

    ####
    stars = h[j].s[starf][flt.Sphere('30 kpc')]
    if (len(stars)<5): return tempdct
    center_avd(stars)
    pynbody.analysis.angmom.faceon(stars,cen_size='31 kpc',disk_size='31 kpc',cen=(0,0,0),vcen=(0,0,0))

    def filt(lst):
        return np.array([1. if x > 0 else 0. for x in lst])

    K = sum(0.5*stars['mass']*stars['v2'])
    K_c = sum(0.5*stars['mass']*(stars['jz']/stars['rxy']*filt(stars['jz']))**2)

    tempdct['kappa_co'] = float(K_c/K)
    ####

    print('end', datbase, galname, i)

    return tempdct



fname = 'data/'+args.datbase+'_'+args.galname+'_'+str(args.idx)+'.json'

result = dosnap([args.datbase, args.idx, args.galname])

f=open(fname,'w')
f.write(json.dumps(result))
f.close()

