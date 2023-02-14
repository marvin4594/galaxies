
import math
import numpy as np
import os
#import pynbody
#import pynbody.array as pa
#import pynbody.filt as f
#import pynbody.units as un
import random

#from pynbody.analysis.interpolate import interpolate2d
from sys import stdout


#bhf = f.LowPass('tform', '0 Gyr')
#starf = pynbody.filt.HighPass('tform', '0 Gyr')





#Creates bins with equal number of points
def equalNbins(N_points,N_bins):
    n = int(N_points/N_bins)
    rest = N_points-n*N_bins

    N_ppb = []
    for i in range(N_bins):
        if (rest>0):
            rest-=1
            ex=1
        else: ex=0
        N_ppb.append(n+ex)

    bins = []
    start=0
    for i in range(N_bins):
        bins.append(range(start,start+N_ppb[i]))
        start += N_ppb[i]

    return bins





# Calculates the luminosity (in magnitudes) of a halo in a specific band, and allows to modify the stellar age
def halo_mag(simstars, band='v', addage=0.0):

    if (len(simstars)==0): return np.nan

    lumfile = "cmdlum.npz"
    if os.path.exists(lumfile):
        lums = np.load(lumfile)
    else:
        raise(IOError, "cmdlum.npz (magnitude table) not found")

    age_star = simstars['age'].in_units('yr')+addage
    metals = np.array(simstars['metals'])
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(lums['ages'])
    metals[np.where(metals < np.min(lums['mets']))] = np.min(lums['mets'])
    metals[np.where(metals > np.max(lums['mets']))] = np.max(lums['mets'])

    age_grid = np.log10(lums['ages'])
    met_grid = lums['mets']
    mag_grid = lums[band]

    output_mags = interpolate2d(metals, np.log10(age_star), met_grid, age_grid, mag_grid)

    try:
        vals = output_mags - 2.5 * \
            np.log10(simstars['massform'].in_units('Msol'))
#    except KeyError, ValueError:
    except:
        vals = output_mags - 2.5 * np.log10(simstars['mass'].in_units('Msol'))

    return -2.5 * np.log10(np.sum(10.0 ** (-0.4 * vals)))





# Calculates the 1-sigma scatter of 'data' when compared to their true values 'func'
def scat(func,data):
    scatter = np.sort(np.abs(func-data))
    i = int(0.683*len(scatter))
    di = 0.683*len(scatter)-i
    if (len(scatter)<2):
        return np.nan
    else:
        return di*scatter[i]+(1.-di)*scatter[i-1]




# Linear function
def fct_lin(x,a,b):
    return a*x+b





def find_sfh(h,t_beg,t_end,bins=100):
    stars = h.star[starf]
    binnorm = 1e-9*bins / (t_end-t_beg)
    tforms = stars['tform'].in_units('Gyr')
#    try:
#        weight = stars['massform'].in_units('Msol') * binnorm
#    except:
    weight = stars['mass'].in_units('Msol') * binnorm
    sfh,sfhbines = np.histogram(tforms, range=(t_beg,t_end), weights=weight, bins=bins)
    sfhtimes = 0.5*(sfhbines[1:]+sfhbines[:-1])
    return sfh,sfhtimes




def dicttofile(dictionary, filename, header=True, keylist=[]):
    if (keylist==[]): keylist = [key for key in dictionary[0]]
    f = open(filename,'w')
    if header:
        for key in keylist: f.write('%-19s  ' %(key))
        f.write('\n')
    for i in range(0,len(dictionary)):
        for key in keylist:
            f.write('%-20s ' %(dictionary[i][key]))
        f.write('\n')
    f.close()



def center_avd(sim):
    pynbody.analysis.halo.center(sim, vel=False)
    cen_size = "1 kpc"
    fr = pynbody.filt.Sphere(cen_size)
    if (len(sim.s[fr]) < 5 and len(sim.d[fr]) < 5 and len(sim.g[fr]) < 5):
        cen_size = h_smooth(sim.d, N=5)
        if (len(sim.s)>=5): cen_size = np.min([cen_size,h_smooth(sim.s, N=5)])
        if (len(sim.g)>=5): cen_size = np.min([cen_size,h_smooth(sim.g, N=5)])
#       cen_size = np.min([h_smooth(sim.s, N=5),h_smooth(sim.d, N=5),h_smooth(sim.g, N=5)])
    pynbody.analysis.halo.vel_center(sim, cen_size=cen_size)

'''
    cont=1
    r_cen = pa.SimArray(1, units='kpc')
    d_r_cen = pa.SimArray(1, units='kpc')
    r_max = pa.SimArray(10, units='kpc')

    while(cont and r_cen <= r_max):
        try:
            pynbody.analysis.halo.center(sim,cen_size = r_cen)
            cont = 0
        except:
            r_cen += d_r_cen
'''



def uvec_rand():
    phi = 2*np.pi*np.random.random()
    u = 2*np.random.random()-1
    v = np.sqrt(1.-u**2)
    return (v*np.cos(phi),v*np.sin(phi),u)


'''
def stellar_vel_disp(sim, naxes=10):

    stars = sim.s[starf]
    center_avd(stars)

    sigma = pa.SimArray(np.full(naxes,0.0), units=stars['vel'].units)
    stellar_half_mass_r2D = pa.SimArray(np.full(naxes,0.0), units=stars['r'].units)

    for i in range(naxes):

        stdout.write("\r%d/%d" %(i+1, naxes))
        stdout.flush()

        stars.rotate_x(360*random.random())
        stars.rotate_y(360*random.random())
        stars.rotate_z(360*random.random())

#        half_r = pynbody.analysis.luminosity.half_light_r(stars)     #return value of this function has the wrong unit
#        half_r = half_mass_r(stars)
        stellar_half_mass_r2D[i] = half_mass_r(stars,cen=False, cylindrical=True)

        ps = pynbody.analysis.profile.Profile(stars, nbins=10, max=stellar_half_mass_r2D[i])

        for j in range(naxes):
            if math.isnan(ps['vz_disp'][j]): ps['vz_disp'][j]=0.0
        sigma[i] = np.sqrt(np.sum((ps['vz_disp']**2)*ps['mass'])/np.sum(ps['mass']))

        #print i, half_mass_r2D[i], sigma[i]

    sigma_avg = np.sum(sigma)/naxes
    sigma_err = np.sqrt(sum((sigma-sigma_avg)**2)/(naxes-1))
    stellar_half_mass_r2D_avg = np.sum(stellar_half_mass_r2D)/naxes
    stellar_half_mass_r2D_err = np.sqrt(sum((stellar_half_mass_r2D-stellar_half_mass_r2D_avg)**2)/(naxes-1))

    print('')
    return sigma_avg, sigma_err, stellar_half_mass_r2D_avg, stellar_half_mass_r2D_err
'''

'''
def half_mass_r(sim, cen=True, cylindrical=False):

    if (cen): center_avd(sim)

    half_mass = sim['mass'].sum() * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'

    X = sorted(zip(sim[coord],sim['mass']), key=lambda pair: pair[0])
    mass=0.0

    for i in range(len(X)):
        mass += X[i][1]
        if (mass >= half_mass): break

    return pa.SimArray(X[i][0],sim[coord].units)
'''

'''
def h_smooth(sim, pos=(0,0,0), N=50):
    if (len(sim) < N): return pa.SimArray(np.nan ,'Mpc')
    r = pa.SimArray(np.sort(np.linalg.norm(sim['pos']-pos,axis=1)),sim['pos'].units)
    if (len(sim) == N): return pa.SimArray(r[N-1],r.units)
    return pa.SimArray(0.5*(r[N]+r[N-1]),r.units)
'''


def kernel(x):
    if (x <= 0):
        w = (21/16.)*(1-0.0454684)
    else:
        u = np.sqrt(x*0.25)
        w = 1-u
        w = w*w
        w = w*w
        w = (21/16.)*w*(1+4*u)
    return w




def kernel2(x):
    u = np.sqrt(x*0.25)
    w = 1-u
    w = w*w
    w = w*w
    w = (21/16.)*w*(1+4*u)
    return w




def mdotbondi(sim, bh, N=50, alpha=100, aDot=0):

    if (len(bh)==0): return np.nan, np.nan, np.nan, np.nan, np.nan

    aDot *= 100*un.a*un.km/(un.s*un.Mpc)

    mdots = []
    rhos = []
    rs_bondi = []
    css = []
    hss = []

    pos2 = [bh['pos'][0][0],bh['pos'][0][1],bh['pos'][0][2]]
    vel2 = [bh['vel'][0][0],bh['vel'][0][1],bh['vel'][0][2]]
    sim['pos'] -= pos2
    sim['vel'] -= vel2

    mass = bh[0]['mass']

    hs = h_smooth(sim.g,bh['pos'][0])
    hhs = hs/2
    filt_smooth = f.LowPass('r', hs)
    gas = sim.g[filt_smooth]

    r2 = gas['r']*gas['r']/(hhs* hhs)
    w = kernel2(r2)
    v2 = gas['v2']

#    v2 = ((gas['vel']-aDot*gas['pos'])*(gas['vel']-aDot*gas['pos'])).sum(axis=1)
    vels2 = v2 + gas['cs']*gas['cs']
    ssum = np.sum(w*gas['mass']/(vels2*np.sqrt(vels2)))
    mdot = 4.0*math.pi*alpha*un.G*un.G*mass*mass*ssum/(math.pi*hhs*hhs*hhs)

    rho = np.sum(w*gas['mass'])/(math.pi*hhs*hhs*hhs)
    cs=np.sum(w*gas['cs'])/sum(w)
    r_bondi = 2*mass*un.G/(cs*cs)
 
    sim['pos'] += pos2
    sim['vel'] += vel2

    return mdot.in_units('Msol yr**-1'), rho.in_units('Msol kpc**-3'), r_bondi.in_units('kpc'), cs.in_units('km s**-1'), hs.in_units('kpc')




def quantity_kernel(sim, pos=(0,0,0), N=50, quantity='rho'):

    if (len(sim.g)==0): return np.nan

    pos2 = [pos[0],pos[1],pos[2]]

    sim['pos'] -= pos2

    hs = h_smooth(sim.g)
    hhs = hs/2
    filt_smooth = f.LowPass('r', hs)
    gas = sim.g[filt_smooth]

    r2 = gas['r']*gas['r']/(hhs* hhs)
    w = kernel2(r2)

    q = np.sum(w*gas['mass']*(gas[quantity]/gas['rho']))/(math.pi*hhs*hhs*hhs)

    sim['pos'] += pos2

    return q, hs




def kravtsov(xmasses, z):
        '''Based on Behroozi+ (2013) return what stellar mass corresponds to the
        halo mass passed in.

        **Usage**

           >>> from pynbody.plot.stars import moster
           >>> xmasses = np.logspace(np.log10(min(totmasshalos)),1+np.log10(max(totmasshalos)),20)
           >>> ystarmasses, errors = moster(xmasses,halo_catalog._halos[1].properties['z'])
           >>> plt.fill_between(xmasses,np.array(ystarmasses)/np.array(errors),
                                                 y2=np.array(ystarmasses)*np.array(errors),
                                                 facecolor='#BBBBBB',color='#BBBBBB')
        '''
        loghm = np.log10(xmasses)
        # from Behroozi et al (2013)
        EPS = -1.642
        EPSpe = 0.133
        EPSme = 0.146

        EPSanu = -0.006
        EPSanupe = 0.113
        EPSanume = 0.361

        EPSznu = 0
        EPSznupe = 0.003
        EPSznume = 0.104

        EPSa = 0.119
        EPSape = 0.061
        EPSame = -0.012

        M1 = 11.514
        M1pe = 0.053
        M1me = 0.009

        M1a = -1.793
        M1ape = 0.315
        M1ame = 0.330

        M1z = -0.251
        M1zpe = 0.012
        M1zme = 0.125

        alpha=-1.779
        AL = alpha
        ALpe = 0.02
        ALme = 0.105

        ALa = 0.731
        ALape = 0.344
        ALame = 0.296

        DEL=4.394
        DELpe = 0.087
        DELme = 0.369

        DELa = 2.608
        DELape = 2.446
        DELame = 1.261

        DELz = -0.043
        DELzpe = 0.958
        DELzme = 0.071

        G=0.547
        Gpe = 0.076
        Gme = 0.012

        Ga = 1.319
        Gape = 0.584
        Game = 0.505

        Gz = 0.279
        Gzpe = 0.256
        Gzme = 0.081

        a = 1.0 / (z + 1.0)
        nu = np.exp(-4 * a ** 2)
        logm1 = M1 + nu * (M1a * (a - 1.0) + M1z * z)
        logeps = EPS + nu * (EPSanu * (a - 1.0) + EPSznu * z) - EPSa * (a - 1.0)
        analpha = AL + nu * ALa * (a - 1.0)
        delta = DEL + nu * DELa * (a - 1.0)
        g = G + nu * (Ga * (a - 1.0) + z * Gz)

        x = loghm - logm1
        f0 = -np.log10(2.0) + delta * np.log10(2.0) ** g / (1.0 + np.exp(1))
        smp = logm1 + logeps + f(x, analpha, delta, g) - f0

        if isinstance(smp, np.ndarray):
                scatter = np.zeros(len(smp))
        scatter = 0.218 - 0.023 * (a - 1.0)

        return 10 ** smp, 10 ** scatter





def moster_2018(xmasses, z):
        '''Based on Moster+ (2018) return what stellar mass corresponds to the
        halo mass passed in.
        '''
        hmp = np.log10(xmasses)
        # from Moster et al (2018)

        fb = 0.182

        if (z<0.3):     #z = 0.1
            m1 = 11.78; r = 0.15; b = 1.78; g = 0.57; ms = 10.85; s0 = 0.16; alpha = 1.00
        elif(z<0.75):    #z = 0.5
            m1 = 11.86; r = 0.18; b = 1.67; g = 0.58; ms = 10.80; s0 = 0.14; alpha = 0.75
        elif (z<1.5):     #z = 1.0
            m1 = 11.98; r = 0.19; b = 1.53; g = 0.59; ms = 10.75; s0 = 0.12; alpha = 0.60
        elif(z<3):     #z = 2.0
            m1 = 11.99; r = 0.19; b = 1.46; g = 0.59; ms = 10.70; s0 = 0.10; alpha = 0.45
        elif(z<6):     #z = 4.0
            m1 = 12.07; r = 0.20; b = 1.36; g = 0.60; ms = 10.60; s0 = 0.06; alpha = 0.35
        else:     #z = 8.0
            m1 = 12.10; r = 0.24; b = 1.30; g = 0.60; ms = 10.40; s0 = 0.02; alpha = 0.30

        smp = hmp + np.log10(2.0 * r) - np.log10((10.0 ** (hmp - m1)) ** (-b) + (10.0 ** (hmp - m1)) **(g))
        sigma = s0 + np.log10((10**hmp/10**ms)**(-alpha) + 1)
        return fb * 10 ** smp, 10 ** sigma





def plotmsmbh(axx):
    x_fit = np.arange(8.6, 12.5, 0.1)
    y_fit = 8.16 + 0.79*(x_fit - 11.0)
    eps = 0.38
    p1=axx.plot(10**x_fit,10**y_fit,'c',label='Sani+11')
    axx.fill_between(10**x_fit,10**(y_fit+eps),y2=10**(y_fit-eps),
                 facecolor='#BBFFFF',color='#BBFFFF',alpha=0.6)

    x_fit = np.arange(9.0, 12.2, 0.1)
    y_fit = 8.69 + 1.17*(x_fit - 11.0)
    eps = 0.28
    p2=axx.plot(10**x_fit,10**y_fit,'m',label='Kormendy+Ho 13')
    axx.fill_between(10**x_fit,10**(y_fit+eps),y2=10**(y_fit-eps),
                 facecolor='#F4D7D9',color='#F4D7D9',alpha=0.6)

#    #x_fit = np.arange(8.0, 13.1, 0.1)
#    y_fit = 7.45 + 1.05*(x_fit - 11.0)
#    eps = 0.55
#    p3=axx.plot(10**x_fit,10**y_fit,color='grey',label='Reines+Volonteri 15')
#    axx.plot(10**x_fit,10**(y_fit-eps),color='grey',ls=':')
#    axx.plot(10**x_fit,10**(y_fit+eps),color='grey',ls=':')

#    #x_fit = np.arange(8.0, 13.1, 0.1)
#    y_fit = 8.2 + 1.12*(x_fit - 11.0)
#    eps = 0.3
#    p4=axx.plot(10**x_fit,10**y_fit,color='lightgrey',label='H\"{a}ring+Rix 04')
#    axx.plot(10**x_fit,10**(y_fit-eps),color='lightgrey',ls=':')
#    axx.plot(10**x_fit,10**(y_fit+eps),color='lightgrey',ls=':')

    return p1+p2





def plotmhms(axx,z=0,plus_delta=True):
    xmasses = np.logspace(8,14,100)
    ystarmasses = xmasses*0.049/0.3175
    axx.plot(xmasses,ystarmasses,'k:')
#####
    xmv_solid = np.logspace(11, 14, 100)
    yms_moster, errors = moster_2018(xmv_solid,z)

    if (z<=0.5): xmv_solid *= 0.85
    if (z>0.5 and z<=1.5): xmv_solid *= 0.93
    if (z>2.5 and z<=3.5): xmv_solid *= 0.95
    if (z>3.5): xmv_solid *= 0.955

    axx.plot(xmv_solid, yms_behroozi, 'c--')

    if (plus_delta):
        delta = 0.28 + 0.14*(np.log10(yms_behroozi)-11)
        delta = [np.minimum(np.maximum(x,0.0),0.3) for x in delta]
        yms_behroozi *= np.power(10.0,delta)
    p2=axx.plot(xmv_solid, yms_behroozi, 'c', label='Behroozi+13')
    axx.fill_between(xmv_solid,yms_behroozi/np.array(errors),
                 y2=np.array(yms_behroozi)*np.array(errors),
                 facecolor='#BBFFFF',color='#BBFFFF',alpha=0.6)
#####
    xmv_solid = np.logspace(10.4, 14, 100)
    yms_moster, errors = pynbody.plot.stars.moster(xmv_solid,z)
    axx.plot(xmv_solid, yms_moster, 'm--')
    if (plus_delta):
        delta = 0.28 + 0.14*(np.log10(yms_moster)-11)
        delta = [np.minimum(np.maximum(x,0.0),0.3) for x in delta]
        yms_moster *= np.power(10.0,delta)
    p3=axx.plot(xmv_solid, yms_moster, 'm', label='Moster+13')
    axx.fill_between(xmv_solid,yms_moster/np.array(errors),
                 y2=np.array(yms_moster)*np.array(errors),
                 facecolor='#F4D7D9',color='#F4D7D9',alpha=0.6)
#####
    return p1+p2+p3


