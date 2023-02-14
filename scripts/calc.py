
import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
#import random as rd
import scipy, scipy.interpolate, scipy.optimize, scipy.stats

from functions import *

import random
rand = random.SystemRandom()



###
##### Functions
###

def fct_dexp(x,a,b,x1,x2,C,D):
    return C*np.exp(-0.5*((x-x1)/a)**2)+D*np.exp(-0.5*((x-x2)/b)**2)

def calc_tcross(time,y_val,bound):
    if(len(time)<2): return np.nan
    assert(len(time)==len(y_val)==len(bound))
    y_val -= np.array(bound)
    if (max(y_val) < 0.0): return np.nan
    for i in range(1,len(y_val)):
        if(y_val[i]*y_val[i-1]<0.0): break
    return time[i] - y_val[i]*(time[i]-time[i-1])/(y_val[i]-y_val[i-1])

def fct_mag_cut(var,a=0.2,b=-0.5,c=1.,d=1.):
    m,z = var
    return a*m+b*pow(z,c)+d

def divpop(pop):
    if (len(pop)==0): return np.nan
    if (len(pop)==1): return pop[0]
    pop = np.array(pop)
    pop_b = np.unique(pop)
    if (len(pop_b)==1): return pop_b[0]
    diff = pop_b[1:len(pop_b)] - pop_b[0:len(pop_b)-1]
    diff = np.append(diff,np.mean(diff))
    bounds = pop_b + 0.5*diff
    ch = []
  
    for bnd in bounds:
        p1 = pop[pop>bnd]
        p2 = pop[pop<bnd]
  
        ch1 = 0.0
        for i in range(len(p1)):
            for j in range(i+1,len(p1)):
                ch1 += (p1[i]-p1[j])**2
        if (len(p1)>0): ch1 /= max(0.5*len(p1)*(len(p1)-1),1)
  
        ch2 = 0.0
        for i in range(len(p2)):
            for j in range(i+1,len(p2)):
                ch2 += (p2[i]-p2[j])**2
        if (len(p2)>0): ch2 /= max(0.5*len(p2)*(len(p2)-1),1)
  
        ch.append(ch1+ch2)
  
    div = bounds[ch.index(min(ch))]
    ch = np.sort(ch)

    return div





def divpop_ml(pop):
    if (len(pop)==0): return np.nan
    if (len(pop)==1): return pop[0]
    pop = np.array(pop)
    pop_b = np.unique(pop)
    if (len(pop_b)==1): return pop_b[0]
    diff = pop_b[1:len(pop_b)] - pop_b[0:len(pop_b)-1]
    diff = np.append(diff,np.mean(diff))
    bounds = pop_b + 0.5*diff
    ch = []

    for bnd in bounds:
        p1 = pop[pop>bnd]
        p2 = pop[pop<bnd]

        mp1 = np.mean(p1)
        mp2 = np.mean(p2)

        ch1 = 0.0
        ch2 = 0.0
        for i in range(len(p1)):
            ch1 += (p1[i]-mp1)**2
        for i in range(len(p2)):
            ch2 += (p2[i]-mp2)**2

        if (len(p1)>0): ch1 /= len(p1)
        if (len(p2)>0): ch2 /= len(p2)
        ch.append(ch1+ch2)

    div = bounds[ch.index(min(ch))]

    return div




def f_lin2(var,a,b,c,d):
    t,m,s = var
    return a*t + b*m + c*s + d


def scat(func,data,shift=False):
    diff = func-data
    if (shift): diff -= np.mean(diff)
    scatter = np.sort(np.abs(diff))
    i = int(0.683*len(scatter))
    di = 0.683*len(scatter)-i
    if (len(scatter)<2):
        return np.nan
    else:
        return di*scatter[i]+(1.-di)*scatter[i-1]





# Calculates the luminosity (in magnitudes) of a halo in a specific band, and allows to modify the stellar age
def halo_mag(simstars, band='v', addage=0.0):

    filt = np.where(np.array(simstars['massform'])>0.0)[0]
    simstars['massform'] = np.array(simstars['massform'])[filt]
    simstars['age'] = np.array(simstars['age'])[filt]
    simstars['metals'] = np.array(simstars['metals'])[filt]

    if (len(simstars['massform'])==0): return np.nan

    if os.path.exists(lumfile):
        lums = np.load(lumfile)
    else:
        raise(IOError, lumfile+" (magnitude table) not found")

    if any(var=='metals' for var in lums.files):
        v_met = 'metals'
    elif any(var=='mets' for var in lums.files):
        v_met = 'mets'

    # convert units to yr:
    age_star = 1e9*np.array(simstars['age']) + addage
    metals = np.array(simstars['metals'])
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(lums['ages'])
    metals[np.where(metals < np.min(lums[v_met]))] = np.min(lums[v_met])
    metals[np.where(metals > np.max(lums[v_met]))] = np.max(lums[v_met])

    age_grid = np.log10(lums['ages'])
    met_grid = lums[v_met]
    mag_grid = lums[band]

#    output_mags = interpolate2d(metals, np.log10(age_star), met_grid, age_grid, mag_grid)
    output_mags = scipy.interpolate.interpn((met_grid, age_grid), mag_grid, (metals, np.log10(age_star)))

#    try:
    vals = output_mags - 2.5 * np.log10(simstars['massform'])
#    except KeyError, ValueError:
#    except:
#        vals = output_mags - 2.5 * np.log10(simstars['mass'])

    return -2.5 * np.log10(np.sum(10.0 ** (-0.4 * vals)))


def fct_sfr_time(t0=1.0,t_ev=5.0,t_end=13.8,DelT=1.0,N=640,met=0.02,a=0.0,b=1e10,e=1.0,t_q=15.,mode=1):
    #UNITS: Gyr, Msol
    time = []
    u_r = []
    dt = (t_end-t0)/(N-1)

    for i in range(N):
        time.append(t0+i*dt)

    idx = (np.abs(np.array(time) - t_ev)).argmin()

    if (mode==2):
        sfr = [a for j in range(N)]
        for j in range(idx,N):
            sfr[j] = (a*np.exp(-(time[j]-t_ev)/DelT)+b)
#        sfr = [(a*np.exp(-(time[j]-t_sb))+b) for j in range(idx,N)]
    else: sfr = [(a*time[j]**e+b) for j in range(N)]

    sfr[0] = 0.0
    sfr[1] = 0.0

    if (mode == 0):
        sfr = [0.0 for j in range(N)]
        sfr[idx] = b

    for i in range(N):
        if (time[i]>t_q): sfr[i]=0.0

    return time, sfr


def fct_ur_time(time=[], sfr=[], met=0.02, q=0.32):

#    if (len(time)==0): time, sfr = fct_sfr_time(t0=t0,t_end=t_end,N=N,a=a,b=b,e=e,t_q=t_q,ep=ep,nsb=nsb)

    #####
    assert(len(time)==len(sfr))
    N = len(time)
#    u_r = []
    dt = [time[i+1]-time[i] for i in range(N-1)]
#    dt.append(dt[-1])

    simstars = [dict() for i in range(N-1)]

    for i in range(N-1):
        simstars[i]['age'] = [time[i]-((1.0-q)*time[j+1]+q*time[j]) for j in range(i)]
        simstars[i]['metals'] = [met for j in range(i)]
        simstars[i]['massform'] = [sfr[j]*dt[j]*1e9 for j in range(i)]

    u_r = [float(halo_mag(simstars[i],band='u')-halo_mag(simstars[i],band='r')) for i in range(N-1)]

    return time[1:], u_r


def fct_bounds(time,sfr):
    dt = np.array(time[1:])-np.array(time[:-1])
    sfr2 = 0.5*(np.array(sfr[1:])+np.array(sfr[:-1]))
    time2 = 0.5*(np.array(time[1:])+np.array(time[:-1]))
    dm = sfr2*dt
    mstar = np.log10([np.sum(dm[:i]) for i in range(len(dm))])
    var = np.vstack((mstar,time2))
    mag_cut_low = fct_mag_cut(var,*data_all['postp']['pb_time']).tolist()
    mag_cut_high = fct_mag_cut(var,*data_all['postp']['pr_time']).tolist()
    return mag_cut_low, mag_cut_high

def fct_tgreen(time,sfr,met=0.02):
    '''
    dt = np.array(time[1:])-np.array(time[:-1])
    sfr2 = 0.5*(np.array(sfr[1:])+np.array(sfr[:-1]))
    time2 = 0.5*(np.array(time[1:])+np.array(time[:-1]))
    dm = sfr2*dt
    mstar = np.log10([np.sum(dm[:i]) for i in range(len(dm))])
    var = np.vstack((mstar,time2))
    mag_cut_low = fct_mag_cut(var,*data_all['postp']['pb_time'])
    mag_cut_high = fct_mag_cut(var,*data_all['postp']['pr_time'])
    '''
    mag_cut_low, mag_cut_high = fct_bounds(time,sfr)
    result = fct_ur_time(time,sfr,met)
    tb = calc_tcross(result[0],result[1],mag_cut_low)
    tr = calc_tcross(result[0],result[1],mag_cut_high)
    return (tr-tb), tr, tb




###
##### options parsing
###

parser = argparse.ArgumentParser(description='calc and plot')
parser.add_argument('-t', '--threads', type=int, help='number of threads',default=1)
parser.add_argument('-c', '--cores', type=int, help='number of cores',default=16)
parser.add_argument('-k', '--tasks', nargs='+', help='what to do', default=['misc','valley','colors','tgreen','quant_tg','sfrfit','collect','lfile_simp','lfile_sfr','lfile_tgreen','lfile_tg_sfon'])
args = parser.parse_args()





#pynbody.config['number_of_threads'] = 1
os.environ['OMP_NUM_THREADS']=str(args.threads)


prfclr = 'bin_'     # 'bin_' or 'sin_'

urcolor = prfclr+'u-r'

nts = 64

Nbins = 4

fac = 1.0

#lumfile = "../aux/cmdlum.npz"
#lumfile = "../aux/bpass_sin_imf135_300.npz"
lumfile = "../aux/bpass_bin_imf135_300.npz"

fname_data_all = '../data/data_all.json'
fname_data_calc = '../data/data_calc.json'

####################
####################


###
##### Calculate crossing times etc
###

if any(task == 'misc' for task in args.tasks):

    f = open(fname_data_all, 'r')
    data_all = json.load(f)
    f.close()

###

    data_all['postp'] = dict()

    galaxies = data_all['galaxies']
    databases = data_all['databases']
###

    for datbase in databases:
        for galname in galaxies[datbase]:
            sfr_min = (data_all[datbase][0][galname]['mstar']/data_all[datbase][0][galname]['nstar'])/(1e9*0.215769)
            data_all[datbase][0][galname]['sfr_min'] = sfr_min
            for i in range(data_all[datbase][0][galname]['n_steps']):
                if (data_all[datbase][i][galname]['sfr_ts'] < sfr_min):
                    data_all[datbase][i][galname]['sfr_ts'] = sfr_min

##### Calculate redshift

    for i in range(nts):
        for datbase in databases:
            for galname in data_all[datbase][i].keys():
                data_all[datbase][i][galname]['redsh'] = (1./data_all[datbase][i][galname]['a_scale']) - 1.0

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'valley' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()


    dtbs = ['nihao_classic','nihao_bh','nihao_ell_bh']

    ### Gather all data
    nts2 = 55
    min_mstar = [min([np.log10(data_all[dtb][i][gname]['mstar']) for dtb in ['nihao_bh','nihao_ell_bh'] for gname in data_all[dtb][i].keys()]) for i in range(nts2)]

    time = [[data_all[dtb][i][gname]['time'] for dtb in dtbs for gname in data_all[dtb][i].keys()] for i in range(nts2)]
    mstar = [[np.log10(data_all[dtb][i][gname]['mstar']) for dtb in dtbs for gname in data_all[dtb][i].keys()] for i in range(nts2)]
    color = [[data_all[dtb][i][gname][urcolor] for dtb in dtbs for gname in data_all[dtb][i].keys()] for i in range(nts2)]

    for i in range(nts2):
        filt = np.where(~np.isnan(time[i]) & ~np.isnan(color[i]) & ~np.isnan(mstar[i]) & (mstar[i]>min_mstar[i]))[0]
        time[i] = np.array(time[i])[filt]
        color[i] = np.array(color[i])[filt]
        mstar[i] = np.array(mstar[i])[filt]

    time = np.concatenate(time).ravel()
    mstar = np.concatenate(mstar).ravel()
    color = np.concatenate(color).ravel()


    ### prepare

    num = len(time)
    group = [i%2 for i in range(num)]
    var = np.vstack((time,mstar,group))
    y = color
    K = 2
    dst = [0]*K

    p = [0,0,3,y[rand.randint(0,num)]]
    p_old = [1,1,1,1]


    ### main loop
    itr = 0
    while(np.equal(p,p_old).all()==False):
        itr += 1
        for i in range(num):
            for k in range(K):
                dst[k] = (y[i]-f_lin2((var[0][i],var[1][i],k),*p))**2
            var[2][i] = dst.index(min(dst))
        p_old = p
        p, pcov = scipy.optimize.curve_fit(f_lin2,var,y)

    f0 = np.where(var[2]==0)[0]
    f1 = np.where(var[2]==1)[0]
 
    e0=scat(f_lin2(var.T[f0].T,*p),y[f0])
    e1=scat(f_lin2(var.T[f1].T,*p),y[f1])

    pb = [p[1],p[0],1.0,p[3]+e0]
    pr = [p[1],p[0],1.0,p[2]+p[3]-e1]
    print(pb,e0)
    print(pr,e1)

    data_all['postp']['pb_time'] = pb
    data_all['postp']['pr_time'] = pr

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()




if any(task == 'valley_old' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    ##### calculate boundaries of green valley
    
    ### gather all data needed to calculate the green valley
    vly_galdata = [dict() for i in range(nts)]

    for i in range(nts):
        # databases to use
        dtbs = ['nihao_classic','nihao_bh','nihao_ell_bh']
        # quantities needed
        qnts = ['mstar',urcolor,'time','redsh']

        # gather these quantities
        for qnt in qnts:
            vly_galdata[i][qnt] = np.array([data_all[dtb][i][gname][qnt] for dtb in dtbs for gname in data_all[dtb][i].keys()])
        vly_galdata[i]['mstar'] = np.log10(vly_galdata[i]['mstar'])

        mstar_bh = [np.log10(data_all[dtb][i][gname]['mstar']) for dtb in ['nihao_bh','nihao_ell_bh'] for gname in data_all[dtb][i].keys()]

        # filter unwanted data
        filt = (~np.isnan(vly_galdata[i][urcolor])) & (vly_galdata[i]['mstar'] > min(mstar_bh))
        for qnt in qnts:
            vly_galdata[i][qnt] = vly_galdata[i][qnt][filt]

        # sort data according to mstar
        X = sorted(zip(vly_galdata[i]['mstar'],vly_galdata[i][urcolor],vly_galdata[i]['time'],vly_galdata[i]['redsh']), key=lambda pair: pair[0])
        for j,qnt in enumerate(qnts):
            vly_galdata[i][qnt] = np.array([X[k][j] for k in range(len(X))])

    # divide the data into mass bins and calculate the red and blue boundary of the green valley for each bin and each redshift
    vly_bindata = dict()
    for clr in ['blue','red']:
        vly_bindata[clr] = dict()
        for qnt in qnts:
            vly_bindata[clr][qnt] = []    

    for i in range(nts):
        mags = vly_galdata[i][urcolor]
        mstar = vly_galdata[i]['mstar']

        div_all = divpop_ml(mags)

        if (len(mags[mags<div_all]) < 2 or len(mags[mags>div_all]) < 2): continue

        bins = equalNbins(len(mstar),Nbins)

        for j in range(Nbins):

            p = mags[bins[j]]
            div = divpop_ml(p)

            pdiv = dict()
            pdiv['blue'] = p[p<div]
            pdiv['red'] = p[p>div]

            bnds = dict()
            bnds['red'] = np.mean(pdiv['red'])-fac*np.std(pdiv['red'])
            bnds['blue'] = np.mean(pdiv['blue'])+fac*np.std(pdiv['blue'])

            if (bnds['red'] < div_all): continue
            if (bnds['blue'] > div_all): continue

            for clr in ['blue','red']:
                if not (len(pdiv[clr])<2):
                    for qnt in ['mstar','time','redsh']:
                        vly_bindata[clr][qnt].append(np.mean(vly_galdata[i][qnt][bins[j]]))
                    vly_bindata[clr][urcolor].append(bnds[clr])

    for clr in ['blue','red']:
        filt = np.where(~np.isnan(vly_bindata[clr][urcolor]) & ~np.isnan(vly_bindata[clr]['mstar']) & ~np.isnan(vly_bindata[clr]['redsh']))[0]
        for qnt in qnts:
            vly_bindata[clr][qnt] = list(np.array(vly_bindata[clr][qnt])[filt])

    data_all['vly_bindata'] = vly_bindata


    # fit the binned data to a function

    x = np.vstack((vly_bindata['blue']['mstar'],vly_bindata['blue']['redsh']))
    y = vly_bindata['blue'][urcolor]
    pg = [0.2,-0.25,0.6,0.4]
    popt_mags_b, pcov = scipy.optimize.curve_fit(fct_mag_cut,x,y,p0=pg)
    x = np.vstack((vly_bindata['red']['mstar'],vly_bindata['red']['redsh']))
    y = vly_bindata['red'][urcolor]
    popt_mags_r, pcov = scipy.optimize.curve_fit(fct_mag_cut,x,y,p0=pg)

    data_all['postp']['popt_mags_b'] = np.ndarray.tolist(popt_mags_b)
    data_all['postp']['popt_mags_r'] = np.ndarray.tolist(popt_mags_r)

    ####

    def fct_mag_comb(var,a=0.2,b=-0.5,c=1.,d=1.,e=1.0):
        m,z,s = var
        return a*m+b*z+d-s*e

    shift_b = np.full(len(vly_bindata['blue']['redsh']),0.0)
    shift_r = np.full(len(vly_bindata['red']['redsh']),1.0)

    shift_d = np.concatenate((shift_b,shift_r))
    mmstar_d = np.concatenate((vly_bindata['blue']['mstar'],vly_bindata['red']['mstar']))
    redsh_d = np.concatenate((vly_bindata['blue']['redsh'],vly_bindata['red']['redsh']))
    col_d = np.concatenate((vly_bindata['blue'][urcolor],vly_bindata['red'][urcolor]))

    pmb, pcov = scipy.optimize.curve_fit(fct_mag_comb,np.vstack((mmstar_d,redsh_d,shift_d)),col_d,p0=[0.2,-0.25,0.6,0.4,1.0])

    data_all['postp']['pb_redsh'] = [pmb[0],pmb[1],1.0,pmb[3]]
    data_all['postp']['pr_redsh'] = [pmb[0],pmb[1],1.0,pmb[3]-pmb[4]]
    ####

    shift_b = np.full(len(vly_bindata['blue']['time']),0.0)
    shift_r = np.full(len(vly_bindata['red']['time']),1.0)

    shift_d = np.concatenate((shift_b,shift_r))
    mmstar_d = np.concatenate((vly_bindata['blue']['mstar'],vly_bindata['red']['mstar']))
    redsh_d = np.concatenate((vly_bindata['blue']['time'],vly_bindata['red']['time']))
    col_d = np.concatenate((vly_bindata['blue'][urcolor],vly_bindata['red'][urcolor]))

    pmb, pcov = scipy.optimize.curve_fit(fct_mag_comb,np.vstack((mmstar_d,redsh_d,shift_d)),col_d,p0=[0.2,-0.25,0.6,0.4,1.0])

    data_all['postp']['pb_time'] = [pmb[0],pmb[1],1.0,pmb[3]]
    data_all['postp']['pr_time'] = [pmb[0],pmb[1],1.0,pmb[3]-pmb[4]]
    print('green valley fit: ',data_all['postp']['pb_time'],data_all['postp']['pr_time'])
    ## TEST
    p = [ 0.06432271, 0.2350446, 0.7863884, -1.45690435]
    data_all['postp']['pb_time'] = [p[1],p[0],1.0,p[3]]
    data_all['postp']['pr_time'] = [p[1],p[0],1.0,p[2]+p[3]]

#    data_all['postp']['pb_time'] = [0.2350446, 0.06432271, 1.0, -1.2335429548442525]
#    data_all['postp']['pr_time'] = [0.2350446, 0.06432271, 1.0, -0.8565105892394012]
    data_all['postp']['pb_time'] = [0.09522965520983862, 0.056087491582878644, 1.0, 0.3257262382162646]
    data_all['postp']['pr_time'] = [0.09522965520983862, 0.056087491582878644, 1.0, 0.8026589971395381]

    ## END TEST
    ####

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'colors' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    # calculate colors for each timestep
    for datbase in data_all['databases']:

        for galname in data_all[datbase][0].keys(): #galaxies[datbase]:

            N = data_all[datbase][0][galname]['n_steps']

            for i in range(N):
                data_all[datbase][i][galname]['color'] = 'x'
                x = data_all[datbase][i][galname][prfclr+'u-r']
                if np.isnan(x): continue
                var = ((np.log10(data_all[datbase][i][galname]['mstar']),data_all[datbase][i][galname]['time']))
                mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
                mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])
                if (x<mag_cut_b):
                    data_all[datbase][i][galname]['color'] = 'b'
                elif (x>mag_cut_r):
                    data_all[datbase][i][galname]['color'] = 'r'
                elif (mag_cut_r>=x>=mag_cut_b):
                    data_all[datbase][i][galname]['color'] = 'g'

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'overview' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    for datbase in data_all['databases']:
        st_b=0
        st_g=0
        st_r=0
        en_b=0
        en_g=0
        en_r=0
        for galname in data_all[datbase][0].keys():
            N = data_all[datbase][0][galname]['n_steps']
            for i in range(N):
                if not np.isnan(data_all[datbase][i][galname]['u-r']): i_last = i
                if (data_all[datbase][i][galname]['color'] == 'b'):
                    if (i==0): st_b+=1
                elif (data_all[datbase][i][galname]['color'] == 'r'):
                    if (i==0): st_r+=1
                elif (data_all[datbase][i][galname]['color'] == 'g'):
                    if (i==0): st_g+=1
            if (data_all[datbase][i_last][galname]['color']=='b'): en_b+=1
            if (data_all[datbase][i_last][galname]['color']=='g'): en_g+=1
            if (data_all[datbase][i_last][galname]['color']=='r'): en_r+=1
#            print(datbase,galname,data_all[datbase][i_last][galname]['color'],data_all[datbase][0][galname]['color'])
            if (datbase=='nihao_bh' or datbase=='nihao_ell_bh'): print('     ',len(data_all[datbase][0][galname]['tg_true']))
        print('datbase: ', datbase, 'len = ', len(data_all[datbase][0].keys()))
        print('datbase: ', datbase, ', z=max, N_b = ', en_b, ', N_g = ', en_g, ', N_r = ', en_r)
        print('datbase: ', datbase, ', z=0, N_b = ', st_b, ', N_g = ', st_g, ', N_r = ', st_r)





new_params = ['tb_simp','tr_simp','tg_simp','tb_true','tr_true','tg_true','tb_sfon','tr_sfon','tg_sfon','age_sfonly','mass_sfonly']

if any(task == 'tgreen' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    # init new param arrays
    for datbase in ['nihao_ell_bh','nihao_bh','nihao_nadine']:
        for galname in data_all[datbase][0].keys(): #galaxies[datbase]:
            for key in new_params: data_all[datbase][0][galname][key] = []


    # calculate tgreen in this loop
    for datbase in ['nihao_ell_bh','nihao_bh','nihao_nadine']:

        for galname in data_all[datbase][0].keys():

            for key in new_params: data_all[datbase][0][galname][key] = []

            N = data_all[datbase][0][galname]['n_steps']

            for i in range(N)[::-1]:

                ##### determine here if galaxy actually crosses the green valley after this timestep
                # select galaxies that are blue now:
                if not (data_all[datbase][i][galname]['color'] == 'b'): continue
                if (i==0): break
                # select gals that actually cross the valley, and don't just return to blue
                n_end = -1
                for j in range(0,i)[::-1]:
                    col = data_all[datbase][j][galname]['color']
                    if (col == 'b' or col == 'x'):
                        break
                    if (col == 'r'):
                        n_end = j
                        break
                if (n_end < 0): continue


                ##### calc tgreen for SF only, no mergers
                time_sfonly = []
                mstar_sfonly = []
                u_r_sfonly = []
                for j in range(i+1):
                    ix =  i-j     # go through all later timesteps i, i-1, i-2 ... 0
                    mag_u = []
                    mag_r = []
                    # take stellar pop at original timestep i, but aged accordingly to timestep ix
#                    if not (prfclr+'u_alt' in data_all[datbase][i][galname]): continue
                    mag_u.append(data_all[datbase][i][galname][prfclr+'u_alt'][j])
                    mag_r.append(data_all[datbase][i][galname][prfclr+'r_alt'][j])
                    for k in range(i-ix):    
                        ixx = i-(k+1)        #go through all timesteps from i-1 to ix
                        mag_u.append(data_all[datbase][ixx][galname][prfclr+'u_new'][ixx-ix])
                        mag_r.append(data_all[datbase][ixx][galname][prfclr+'r_new'][ixx-ix])
                    time_sfonly.append(data_all[datbase][ix][galname]['time'])
                    mstar_sfonly.append(data_all[datbase][ix][galname]['mstar'])
                    u_r_sfonly.append((-2.5 * np.log10(np.sum(10.0 ** (-0.4 * np.array(mag_u))))) - \
                                      (-2.5 * np.log10(np.sum(10.0 ** (-0.4 * np.array(mag_r))))))

                var = np.vstack((np.log10(mstar_sfonly),time_sfonly))
                mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
                mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

                tr_sfon = calc_tcross(time_sfonly,u_r_sfonly,mag_cut_r)
                tb_sfon = calc_tcross(time_sfonly,u_r_sfonly,mag_cut_b)
                tg_sfon = tr_sfon-tb_sfon
#                print('SFON', galname, tg_sfon)

                ## calc avg age and tot mass of new stars
                t_form = []
                mass_sfonly = 0.
#                for k in range(1,n_end+1):
                for k in range(1,i):
                    ix = i-k
                    if (data_all[datbase][ix][galname]['time']>tr_sfon): break
                    tform = data_all[datbase][ix][galname]['time'] - data_all[datbase][ix][galname]['age_new']
                    t_form.append(tform*data_all[datbase][ix][galname]['mass_new'])
                    mass_sfonly += data_all[datbase][ix][galname]['mass_new']
                if (mass_sfonly == 0.0): t_form_sfonly = np.nan
                else: t_form_sfonly = np.sum(t_form)/mass_sfonly
                age_sfonly = tr_sfon-t_form_sfonly

                ##### calc tgreen for simplified luminosity curves:
                time_alt = data_all[datbase][i][galname]['time_alt']
                color_alt = data_all[datbase][i][galname][prfclr+'u-r_alt']
#                if not (isinstance(time_alt,list)): continue
#                if not (isinstance(color_alt,list)): continue
#                if (len(time_alt)<2): continue

                ms = [np.log10(data_all[datbase][j][galname]['mstar']) for j in range(i,i-len(time_alt),-1)]
                rs = [data_all[datbase][j][galname]['time'] for j in range(i,i-len(time_alt),-1)]
                var = np.vstack((ms,rs))
                mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
                mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

                tr_simp = calc_tcross(time_alt,color_alt,mag_cut_r)
                tb_simp = calc_tcross(time_alt,color_alt,mag_cut_b)
                tg_simp = tr_simp-tb_simp
#                print('SIMP', galname, tg_simp)
#                if np.isnan(tg_simp): continue

                time = np.array([data_all[datbase][j][galname]['time'] for j in range(N)])


                ##### calculate real tgreen (for actual luminosity curves):

                time = [data_all[datbase][j][galname]['time'] for j in range(n_end,i+1)]
                color = [data_all[datbase][j][galname][prfclr+'u-r'] for j in range(n_end,i+1)]
                mstar = [data_all[datbase][j][galname]['mstar'] for j in range(n_end,i+1)]

                var = np.vstack((np.log10(mstar),time))
                mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
                mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

                tr_true = calc_tcross(time,color,mag_cut_r)
                tb_true = calc_tcross(time,color,mag_cut_b)
                tg_true = tr_true-tb_true
#                print('TRUE', galname, tg_true)
#                print('EXSI', galname, tg_true-tg_sfon)
                ##### store all data in json database 
#                print(datbase,galname,tg_true,tb_true,tr_true)
                data_all[datbase][i][galname]['time_sfonly'] = time_sfonly
                data_all[datbase][i][galname][prfclr+'u_r_sfonly'] = u_r_sfonly

                data_all[datbase][0][galname]['mass_sfonly'].append(mass_sfonly)
                data_all[datbase][0][galname]['age_sfonly'].append(age_sfonly)

                data_all[datbase][0][galname]['tb_sfon'].append(tb_sfon)
                data_all[datbase][0][galname]['tr_sfon'].append(tr_sfon)
                data_all[datbase][0][galname]['tg_sfon'].append(tg_sfon)

                data_all[datbase][0][galname]['tb_simp'].append(tb_simp)
                data_all[datbase][0][galname]['tr_simp'].append(tr_simp)
                data_all[datbase][0][galname]['tg_simp'].append(tg_simp)

                data_all[datbase][0][galname]['tb_true'].append(tb_true)
                data_all[datbase][0][galname]['tr_true'].append(tr_true)
                data_all[datbase][0][galname]['tg_true'].append(tg_true)


    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'quant_tg' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    quants = ['mstar','star_age_mean','star_age_median','star_age_std','star_metals_mean','star_metals_median','star_metals_std']

    for datbase in ['nihao_ell_bh','nihao_bh']:
        for galname in data_all[datbase][0].keys():

            N = data_all[datbase][0][galname]['n_steps']
            time = np.array([data_all[datbase][i][galname]['time'] for i in range(N)])

            for quant in quants:
                data_all[datbase][0][galname][quant+'_tb'] = []
                qnt = np.log10([data_all[datbase][i][galname][quant] for i in range(N)])
                f_qnt = scipy.interpolate.interp1d(time[~np.isnan(qnt)],qnt[~np.isnan(qnt)],fill_value='extrapolate')
                data_all[datbase][0][galname][quant+'_tb'] = [10**float(f_qnt(tb)) for tb in data_all[datbase][0][galname]['tb_simp']]

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'sfrfit' for task in args.tasks):

    def strictly_increasing(L):
        return all(x<y for x, y in zip(L, L[1:]))

    def getrange(tb,tr,time):
        for i in range(len(time)):
            if (tb>time[i]): break
        i_b = i+1 
        for i in range(len(time)):
            if (tr>time[i]): break
        i_r = i-1
        return range(i_r,i_b)      


    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    for datbase in ['nihao_ell_bh','nihao_bh','nihao_nadine']:
        for galname in data_all[datbase][0].keys(): #galaxies[datbase]:

            tb = data_all[datbase][0][galname]['tb_sfon']
            tr = data_all[datbase][0][galname]['tr_sfon']

            N = data_all[datbase][0][galname]['n_steps']
            time = np.array([data_all[datbase][i][galname]['time'] for i in range(N)])
            sfr = np.array([data_all[datbase][i][galname]['sfr_ts'] for i in range(N)])
            mstar = np.array([data_all[datbase][i][galname]['mstar'] for i in range(N)])

            for i in range(N):
                data_all[datbase][i][galname]['time_fit'] = np.nan
                data_all[datbase][i][galname]['sfr_fit'] = np.nan
            data_all[datbase][0][galname]['dt_dec'] = []
            data_all[datbase][0][galname]['sfr_mon'] = []

            for i in range(len(tb)):
                if (np.isnan(tr[i])): tr[i] = 13.87
                rnge = getrange(tb[i],tr[i],time)
                time_fit = time[rnge]
                sfr_fit = sfr[rnge]
                data_all[datbase][0][galname]['sfr_mon'].append(strictly_increasing(sfr_fit))

                popt, pcov = scipy.optimize.curve_fit(fct_lin,time_fit,np.log(sfr_fit),p0=[-1.,1.])
                sfr_fit = np.exp(fct_lin(time_fit,*popt))
                data_all[datbase][0][galname]['dt_dec'].append(-1.0/popt[0])

                for i in range(len(rnge)):
                    data_all[datbase][rnge[i]][galname]['time_fit'] = time_fit[i]
                    data_all[datbase][rnge[i]][galname]['sfr_fit'] = sfr_fit[i]

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'collect' for task in args.tasks):

    dbases = ['nihao_ell_bh','nihao_bh']
    new_params += ['dt_dec','sfr_mon','mstar_tb','star_age_mean_tb','star_age_median_tb','star_age_std_tb','star_metals_mean_tb','star_metals_median_tb','star_metals_std_tb']

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    for prm in new_params:
        data_all['postp'][prm] = [data_all[datbase][0][galname][prm][i] for datbase in dbases \
             for galname in data_all[datbase][0].keys() for i in range(len(data_all[datbase][0][galname][prm]))]

    data_all['postp']['mstar_z0'] = [data_all[datbase][0][galname]['mstar'] for datbase in dbases \
             for galname in data_all[datbase][0].keys() for i in range(len(data_all[datbase][0][galname]['tg_true']))]
    data_all['postp']['galname'] = [galname for datbase in dbases \
             for galname in data_all[datbase][0].keys() for i in range(len(data_all[datbase][0][galname]['tg_true']))]

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'lfile_simp' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    tgreen_ages = []
    tgreen_mets = []
    tgreen_mstar = []
    ages = []
    mstar_tb = []

    sfr_const = 6e10
    tq = 5.

    tsbs = [1.1,3.,5.,7.,9.,11.,13.]
    mets = [0.002,0.003,0.004,0.007,0.01,0.02,0.03,0.04]
    sfrs = [1e9,2e10,3e11]

    for tsb in tsbs:
        tgr, tr, tb = fct_tgreen(*fct_sfr_time(t_q=tsb,b=sfr_const))
        tgreen_ages.append(tr-tb)
        ages.append(tb-0.5*tsb)

    for met in mets:
        tgr, tr, tb = fct_tgreen(*fct_sfr_time(t_q=tq,b=sfr_const),met=met)
        tgreen_mets.append(tr-tb)

    for sfr_const in sfrs:
        tgr, tr, tb = fct_tgreen(*fct_sfr_time(t_q=tq,b=sfr_const))
        mstar_tb.append(np.log10(sfr_const*tb))
        tgreen_mstar.append(tr-tb)

    data_all['postp']['lfile_simp_ages'] = ages
    data_all['postp']['lfile_simp_mets'] = mets
    data_all['postp']['lfile_simp_mstar'] = mstar_tb

    data_all['postp']['lfile_simp_tg_ages'] = tgreen_ages
    data_all['postp']['lfile_simp_tg_mets'] = tgreen_mets
    data_all['postp']['lfile_simp_tg_mstar'] = tgreen_mstar

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'lfile_sfr' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    data_all['lfile'] = dict()

    t, ur = fct_ur_time(*fct_sfr_time())
    data_all['lfile']['time'] = t
    data_all['lfile']['sfr_const'] = ur

    time, sfr = fct_sfr_time(a=1.e10,b=0.,mode=2)
    t, ur = fct_ur_time(time,sfr)
    ml, mh = fct_bounds(time,sfr)
    data_all['lfile']['sfr_exp'] = ur
    data_all['lfile']['sfr_exp_ml'] = ml
    data_all['lfile']['sfr_exp_mh'] = mh

    t, ur = fct_ur_time(*fct_sfr_time(a=1.,e=-2.,b=0.))
    data_all['lfile']['sfr_t-2'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(mode=0,t_ev=4.))
    data_all['lfile']['sfr_SB'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(t_q=5.))
    data_all['lfile']['sfr_tq'] = ur



    t, ur = fct_ur_time(*fct_sfr_time(a=1.,b=0.,mode=2,DelT=2.))
    data_all['lfile']['sfr_dt2'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(a=1.,b=0.,mode=2,DelT=1.))
    data_all['lfile']['sfr_dt1'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(a=1.,b=0.,mode=2,DelT=0.5))
    data_all['lfile']['sfr_dt05'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(a=1.,b=0.,mode=2,DelT=0.2))
    data_all['lfile']['sfr_dt02'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(a=1.,b=0.,mode=2,DelT=0.1))
    data_all['lfile']['sfr_dt01'] = ur


    t, ur = fct_ur_time(*fct_sfr_time(t0=1.,a=1.,b=0.,mode=2,t_ev=1.))
    data_all['lfile']['sfr_tq1'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(t0=1.,a=1.,b=0.,mode=2,t_ev=3.))
    data_all['lfile']['sfr_tq3'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(t0=1.,a=1.,b=0.,mode=2,t_ev=5.))
    data_all['lfile']['sfr_tq5'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(t0=1.,a=1.,b=0.,mode=2,t_ev=10.))
    data_all['lfile']['sfr_tq10'] = ur
    t, ur = fct_ur_time(*fct_sfr_time(t0=1.,a=1.,b=0.,mode=2,t_ev=20.))
    data_all['lfile']['sfr_tq20'] = ur


    t, ur = fct_ur_time(time,sfr,met=0.01)
    data_all['lfile']['sfr_met1'] = ur
    t, ur = fct_ur_time(time,sfr,met=0.02)
    data_all['lfile']['sfr_met2'] = ur
    t, ur = fct_ur_time(time,sfr,met=0.03)
    data_all['lfile']['sfr_met3'] = ur
    t, ur = fct_ur_time(time,sfr,met=0.04)
    data_all['lfile']['sfr_met4'] = ur


    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'lfile_tg_sfon' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    tg = []
    dM = []
    age = []

    a = 6e10
    t_ev = 5.

    for DelT in [0.1,0.3,0.5,1.0,1.6]:
        tgr, tr, tb = fct_tgreen(*fct_sfr_time(a=a,b=0.,t_ev=t_ev,DelT=DelT,mode=2))
        tg.append(tgr)
        dM.append(a*DelT*(np.exp(-(tb-t_ev)/DelT)-np.exp(-(tr-t_ev)/DelT)))
        tform = (a*DelT/dM[-1])*((DelT+tb)*np.exp(-(tb-t_ev)/DelT)-(DelT+tr)*np.exp(-(tr-t_ev)/DelT))
        age.append(tr-tform)

    data_all['lfile']['sfon_tg'] = tg
    data_all['lfile']['sfon_dM'] = dM
    data_all['lfile']['sfon_age'] = age

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()





if any(task == 'lfile_tgreen' for task in args.tasks):

    f = open(fname_data_calc, 'r')
    data_all = json.load(f)
    f.close()

    a = 6e10

    data_all['lfile']['dts'] = [0.1,0.6,0.8,1.1,1.6,2.1,2.6,3.1]
    data_all['lfile']['tqs'] = [0.1,2.1,4.1,6.1,8.1,10.1]
    data_all['lfile']['mets'] = [0.01,0.02,0.03,0.04]
    data_all['lfile']['tgreen_dt'] = [fct_tgreen(*fct_sfr_time(a=a,b=0.,mode=2,DelT=dt))[0] for dt in data_all['lfile']['dts']]
    data_all['lfile']['tgreen_tq'] = [fct_tgreen(*fct_sfr_time(a=a,b=0.,mode=2,t_ev=tq))[0] for tq in data_all['lfile']['tqs']]
    data_all['lfile']['tgreen_met'] = [fct_tgreen(*fct_sfr_time(a=a,b=0.,mode=2),met=met)[0] for met in data_all['lfile']['mets']]

    f=open(fname_data_calc,'w')
    f.write(json.dumps(data_all))
    f.close()


