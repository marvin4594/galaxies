
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.interpolate, scipy.optimize, scipy.stats

from functions import *





plots_paper = ['mag_mstar','mag_time_tracks_comp','mag_time_tracks_comp2','mag_time_tracks_nad','lfile_tgreen','lfile_sfr','timescales','hist_times','tg_simp','tg_sfon']
plots_others = ['mag_time_tracks','vly_redsh','vly_time','times_nad','lfile_sfr_old','time_mstar']

### options parsing

parser = argparse.ArgumentParser(description='calc and plot')
parser.add_argument('-t', '--pictype', type=str, help='file type',default='png')
parser.add_argument('-p', '--plots', nargs='+', help='what to plot',default=plots_paper)
args = parser.parse_args()





def fct_mag_cut(var,a=0.2,b=-0.5,c=1.,d=1.):
    m,z = var
    return a*m+b*pow(z,c)+d


def getrange(tb,tr,time):
    for i in range(len(time)):
        if (tb>time[i]): break
    i_b = i+1
    for i in range(len(time)):
        if (tr>time[i]): break
    i_r = i-1
    return range(i_r,i_b)



colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

prfclr = 'bin_'

urcolor = prfclr+'u-r'

w_halfpage = 6



f = open('../data/data_calc.json', 'r')
data_all = json.load(f)
f.close()

galaxies = data_all['galaxies']
databases = data_all['databases']

np.set_printoptions(precision=3)
print('Green Valley fit:')
print('    ',np.array(data_all['postp']['pb_time']))
print('    ',np.array(data_all['postp']['pr_time']))


###
####
##### PLOT EVERYTHING
####
###

print('plotting')

##### plot color vs mstar

if any(task == 'vly_redsh' for task in args.plots):

    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    x = data_all['vly_bindata']['blue']['redsh']
    y = data_all['vly_bindata']['blue'][urcolor]
    ax.scatter(x,y,marker='o',s=5,color='b')

    x = data_all['vly_bindata']['red']['redsh']
    y = data_all['vly_bindata']['red'][urcolor]
    ax.scatter(x,y,marker='o',s=5,color='r')

    pb = data_all['postp']['pb_redsh']
    pr = data_all['postp']['pr_redsh']

    z = np.arange(0,2,0.1)
    var = np.vstack((np.full(len(z),11.),z))
    y = fct_mag_cut(var,*pb)
    ax.plot(z,y,color='b')
    y = fct_mag_cut(var,*pr)
    ax.plot(z,y,color='r')

    ax.set_ylabel('U-R')
    ax.set_xlabel('redshift')

    plt.savefig('../plots/vly_redsh.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'vly_time' for task in args.plots):

    plt.clf()
    fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

    x = data_all['vly_bindata']['blue']['time']
    y = data_all['vly_bindata']['blue'][urcolor]
    ax.scatter(x,y,marker='o',s=5,color='b')

    x = data_all['vly_bindata']['red']['time']
    y = data_all['vly_bindata']['red'][urcolor]
    ax.scatter(x,y,marker='o',s=5,color='r')

    t = np.arange(3,14,0.1)
    var = np.vstack((np.full(len(t),11.),t))
    y = fct_mag_cut(var,*data_all['postp']['pb_time'])
    ax.plot(t,y,color='b')
    y = fct_mag_cut(var,*data_all['postp']['pr_time'])
    ax.plot(t,y,color='r')

    ax.set_ylabel('U-R')
    ax.set_xlabel('time [Gyr]')

    plt.savefig('../plots/vly_time.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'mag_mstar' for task in args.plots):

    i = 0

    mstar_bh = [np.log10(data_all[dtb][i][gname]['mstar']) for dtb in ['nihao_bh','nihao_ell_bh'] for gname in data_all[dtb][i].keys()]
    color_bh = [data_all[dtb][i][gname][prfclr+'u-r'] for dtb in ['nihao_bh','nihao_ell_bh'] for gname in data_all[dtb][i].keys()]

    gals_yes = []
    gals_no = []
    for galname in data_all['nihao_classic'][i].keys():
        if (np.log10(data_all['nihao_classic'][i][galname]['mstar']) > min(mstar_bh)): gals_yes.append(galname)
        else: gals_no.append(galname)


    plt.clf()
    fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

    ax.scatter(mstar_bh,color_bh,marker='o',s=10,color=colors[1],label='NIHAO BH')

    x = np.log10(np.array([data_all['nihao_classic'][i][galname]['mstar'] for galname in gals_yes]))
    y = np.array([data_all['nihao_classic'][i][galname][prfclr+'u-r'] for galname in gals_yes])
    ax.scatter(x,y,marker='o',s=10,color=colors[0], label='NIHAO classic (used for GV fit)')

    x = np.log10(np.array([data_all['nihao_classic'][i][galname]['mstar'] for galname in gals_no]))
    y = np.array([data_all['nihao_classic'][i][galname][prfclr+'u-r'] for galname in gals_no])
    ax.scatter(x,y,marker='o',s=10,edgecolors=colors[0],facecolors='none', label='NIHAO classic (not used for GV fit)')

    '''
    x = np.log10(np.array([data_all['nihao_bh'][i][galname]['mstar'] for galname in data_all['nihao_bh'][i].keys()]))
    y = np.array([data_all['nihao_bh'][i][galname][prfclr+'u-r'] for galname in data_all['nihao_bh'][i].keys()])
    ax.scatter(x,y,marker='o',s=10,color=colors[3],label='NIHAO BH')

    x = np.log10(np.array([data_all['nihao_ell_bh'][i][galname]['mstar'] for galname in data_all['nihao_ell_bh'][i].keys()]))
    y = np.array([data_all['nihao_ell_bh'][i][galname][prfclr+'u-r'] for galname in data_all['nihao_ell_bh'][i].keys()])
    ax.scatter(x,y,marker='o',s=10,color=colors[3])
    '''

    m = [min(mstar_bh),max(mstar_bh)]
    var = np.vstack((m,[13.8,13.8]))
    ax.plot(m,fct_mag_cut(var,*data_all['postp']['pb_time']),color='b')
    ax.plot(m,fct_mag_cut(var,*data_all['postp']['pr_time']),color='r')

    ax.set_xlim([9,12])
    ax.set_ylabel('U-R')
    ax.set_xlabel('log $M_{\star}$ [M$_\odot$]')

    ax.legend()

    plt.savefig('../plots/mag_mstar.'+args.pictype,dpi=1000,bbox_inches='tight')






##### Plot color and SFR vs time for all galaxies

data_pl = dict()
#data_pl['nihao_ell_bh'] = galaxies['nihao_ell_bh']   #['g6.86e12']   #['g1.55e12','g2.37e12','g2.71e12','g6.86e12','g7.55e12']
#data_pl['nihao_bh'] = ['g2.79e12']
#data_pl['nihao_classic'] = galaxies['nihao_classic'] #['g3.21e11']
#data_pl['nihao_nadine'] = galaxies['nihao_nadine']

data_pl['nihao_ell_bh'] = ['g1.44e13']

if any(task == 'mag_time_tracks' for task in args.plots):

    for datbase in data_pl:
        for galname in data_pl[datbase]: #galaxies[datbase]:

            N = data_all[datbase][0][galname]['n_steps']
            time = [data_all[datbase][i][galname]['time'] for i in range(N)]
            mstar = [np.log10(data_all[datbase][i][galname]['mstar']) for i in range(N)]
            color = [data_all[datbase][i][galname][prfclr+'u-r'] for i in range(N)]
            sfr = [np.log10(data_all[datbase][i][galname]['sfr_ts']) for i in range(N)]

#            if ('time_fit' in data_all[datbase][0][galname]):
#                time_fit = [data_all[datbase][i][galname]['time_fit'] for i in range(N)]
#                sfr_fit = [np.log10(data_all[datbase][i][galname]['sfr_fit']) for i in range(N)]
#                ax.plot(time_fit,sfr_fit,'-o',ms=2,color=colors[3],label='sfr_fit')

            var = np.vstack((mstar,time))
            mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
            mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])


            plt.clf()
            fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

            ax.plot(time,color,'-o',ms=2,color=colors[0],label='U-R')
            ax.plot(time,sfr,'-o',ms=2,color=colors[1],label='SFR')
#            ax.plot(time_fit,sfr_fit,'-o',ms=2,color=colors[3],label='sfr_fit')

            for i in range(N):
                if ('time_sfonly' in data_all[datbase][i][galname]):
                    time_sfonly = data_all[datbase][i][galname]['time_sfonly']
                    u_r_sfonly = data_all[datbase][i][galname][prfclr+'u_r_sfonly']
                    for j in range(len(time_sfonly)):
                        if (u_r_sfonly[j] > mag_cut_r[i-j]): break
                    ax.plot(time_sfonly[:j+1],u_r_sfonly[:j+1],'-',ms=2,color=colors[4])

                    time_alt = data_all[datbase][i][galname]['time_alt']
                    color_alt = data_all[datbase][i][galname][prfclr+'u-r_alt']
                    for j in range(len(time_alt)):
                        if (color_alt[j] > mag_cut_r[i-j]): break
                    ax.plot(time_alt[:j+1],color_alt[:j+1],'-',ms=2,color=colors[2])

            ax.plot(time,mag_cut_b,color='b')
            ax.plot(time,mag_cut_r,color='r')

            ax.legend(loc='lower right',fontsize='xx-small')

            ax.set_xlim([0,14])
#            ax.set_ylim([-2.7,2.2])
            ax.set_ylabel('U-R   log SFR [M$_\odot$ yr$^{-1}$]')
            ax.set_xlabel('time [Gyr]')

            plt.savefig('../plots/mag_time_tracks_'+datbase+'_'+galname+'.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'mag_time_tracks_comp' for task in args.plots):

    gals = ['g1.57e13','g6.70e12','g2.02e13','g2.37e13','g2.79e12','g1.26e12']
    dtbs = ['nihao_ell_bh','nihao_ell_bh','nihao_ell_bh','nihao_ell_bh','nihao_bh','nihao_ell_bh']

    plt.clf()
    fig = plt.figure(figsize=(2*w_halfpage,2.5*w_halfpage))
    ax = fig.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
    #ax.tick_params(axis='x', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #ax.tick_params(axis='y', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    plt.subplots_adjust(hspace=0.001,wspace=0.001)

    axes = [ax1,ax2,ax3,ax4,ax5,ax6]
    datbase = 'nihao_ell_bh'
    print('mag_time_tracks_comp')

    for i, axx in enumerate(axes):
        datbase = dtbs[i]
        galname = gals[i]

        print('    ',galname,["%.2f" % elem for elem in data_all[datbase][0][galname]['tg_true']])

        N = data_all[datbase][0][galname]['n_steps']
        time = [data_all[datbase][i][galname]['time'] for i in range(N)]
        mstar = [data_all[datbase][i][galname]['mstar'] for i in range(N)]
        color = [data_all[datbase][i][galname][prfclr+'u-r'] for i in range(N)]
        sfr = [np.log10(data_all[datbase][i][galname]['sfr_ts']) for i in range(N)]

        var = np.vstack((np.log10(mstar),time))
        mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
        mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

        axx.plot(time,color,'-o',ms=2,color=colors[4],label='U-R')
        axx.plot(time,sfr,'-o',ms=2,color=colors[1],label='SFR')

        axx.plot(time,mag_cut_b,color='b')
        axx.plot(time,mag_cut_r,color='r')
        axx.text(0.03,0.90,galname,transform=axx.transAxes,fontsize='x-large')

        axx.set_xlim([0,13.9])
        axx.set_ylim([-0.45,2.9])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax2.set_yticks([])
    ax4.set_yticks([])
    ax6.set_yticks([])

    ax3.set_ylabel('U-R   log SFR [M$_\odot$ yr$^{-1}$]')
    ax5.set_xlabel('time [Gyr]')
    ax6.set_xlabel('time [Gyr]')
    ax6.legend(loc='lower right')

    plt.savefig('../plots/mag_time_tracks_comp.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'mag_time_tracks_classic' for task in args.plots):

    gals = ['g5.59e09','g1.50e10','g9.59e10','g1.08e11','g1.37e11','g2.57e11']
    dtbs = ['nihao_classic','nihao_classic','nihao_classic','nihao_classic','nihao_classic','nihao_classic']

    plt.clf()
    fig = plt.figure(figsize=(2*w_halfpage,2.5*w_halfpage))
    ax = fig.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
    #ax.tick_params(axis='x', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #ax.tick_params(axis='y', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    plt.subplots_adjust(hspace=0.001,wspace=0.001)

    axes = [ax1,ax2,ax3,ax4,ax5,ax6]
    datbase = 'nihao_ell_bh'
    print('mag_time_tracks_classic')

    for i, axx in enumerate(axes):
        datbase = dtbs[i]
        galname = gals[i]

        N = data_all[datbase][0][galname]['n_steps']
        time = [data_all[datbase][i][galname]['time'] for i in range(N)]
        mstar = [data_all[datbase][i][galname]['mstar'] for i in range(N)]
        color = [data_all[datbase][i][galname][prfclr+'u-r'] for i in range(N)]
        sfr = [np.log10(data_all[datbase][i][galname]['sfr_ts']) for i in range(N)]

        var = np.vstack((np.log10(mstar),time))
        mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
        mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

        axx.plot(time,color,'-o',ms=2,color=colors[0],label='U-R')
        axx.plot(time,sfr,'-o',ms=2,color=colors[1],label='SFR')

        axx.plot(time,mag_cut_b,color='b')
        axx.plot(time,mag_cut_r,color='r')
        axx.text(0.03,0.90,galname,transform=axx.transAxes,fontsize='x-large')

        axx.set_xlim([0,13.9])
#        axx.set_ylim([-0.45,2.9])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax2.set_yticks([])
    ax4.set_yticks([])
    ax6.set_yticks([])

    ax3.set_ylabel('U-R   log SFR [M$_\odot$ yr$^{-1}$]')
    ax5.set_xlabel('time [Gyr]')
    ax6.set_xlabel('time [Gyr]')
    ax6.legend(loc='lower right')

    plt.savefig('../plots/mag_time_tracks_classic.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'mag_time_tracks_comp2' for task in args.plots):

    plt.clf()
    fig = plt.figure(figsize=(2*w_halfpage,2.5*w_halfpage))
    ax = fig.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
    #ax.tick_params(axis='x', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    #ax.tick_params(axis='y', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    plt.subplots_adjust(hspace=0.22,wspace=0.15)

    axes = [ax1,ax2,ax3,ax4,ax5,ax6]

    datbase = 'nihao_ell_bh'

    gals = ['g1.27e12','g2.79e12','g6.70e12','g6.86e12','g1.33e13','g2.07e13']
    dtbs = ['nihao_ell_bh','nihao_bh','nihao_ell_bh','nihao_ell_bh','nihao_ell_bh','nihao_ell_bh']

    t_lim = [[5.8,6.6],[7.5,9.4],[7.6,12.5],[5.8,7.1],[3.5,4.1],[4.1,5.1]]
    c_lim = [[-0.9,2.6],[0.2,2.6],[-0.2,2.8],[-0.4,2.5],[-1.4,2.6],[-0.8,2.4]]

    print('mag_time_tracks_comp2 ... true ... sfon ... simp')

    for i, axx in enumerate(axes):
        datbase = dtbs[i]
        galname = gals[i]

        print('    ',galname,["%.2f" % elem for elem in data_all[datbase][0][galname]['tg_true']],["%.2f" % elem for elem in data_all[datbase][0][galname]['tg_sfon']],["%.2f" % elem for elem in data_all[datbase][0][galname]['tg_simp']])

        N = data_all[datbase][0][galname]['n_steps']
        time = [data_all[datbase][j][galname]['time'] for j in range(N)]
        mstar = [data_all[datbase][j][galname]['mstar'] for j in range(N)]
        color = [data_all[datbase][j][galname][prfclr+'u-r'] for j in range(N)]
        sfr = [np.log10(data_all[datbase][j][galname]['sfr_ts']) for j in range(N)]

        var = np.vstack((np.log10(mstar),time))
        mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
        mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])

        axx.plot(time,color,'-o',lw=2,ms=2,color=colors[4],label='U-R')
        axx.plot(time,sfr,'-o',ms=2,color=colors[1],label='SFR')

        time_fit = [data_all[datbase][i][galname]['time_fit'] for i in range(N)]
        sfr_fit = [np.log10(data_all[datbase][i][galname]['sfr_fit']) for i in range(N)]
        axx.plot(time_fit,sfr_fit,'-o',ms=2,color=colors[6],label='SFR fit')

        for j in range(N):
            if ('time_sfonly' in data_all[datbase][j][galname]):
                time_sfonly = data_all[datbase][j][galname]['time_sfonly']
                u_r_sfonly = data_all[datbase][j][galname][prfclr+'u_r_sfonly']
                for k in range(len(time_sfonly)):
                    if (u_r_sfonly[k] > mag_cut_r[j-k]): break
                axx.plot(time_sfonly[:k+2],u_r_sfonly[:k+2],'-',lw=1.5,color=colors[5],label='SF-only')

                time_alt = data_all[datbase][j][galname]['time_alt']
                color_alt = data_all[datbase][j][galname][prfclr+'u-r_alt']
                for k in range(len(time_alt)):
                    if (color_alt[k] > mag_cut_r[j-k]): break
                axx.plot(time_alt[:k+2],color_alt[:k+2],'-',lw=1,color=colors[2],label='simp')

        axx.plot(time,mag_cut_b,color='b')
        axx.plot(time,mag_cut_r,color='r')

        axx.text(0.7,0.03,galname,transform=axx.transAxes,fontsize='x-large')

        axx.set_xlim(t_lim[i])
        axx.set_ylim(c_lim[i])

    ax3.set_ylabel('U-R   log SFR [M$_\odot$ yr$^{-1}$]')
    ax5.set_xlabel('time [Gyr]')
    ax6.set_xlabel('time [Gyr]')
    ax3.legend(loc='lower left')

    plt.savefig('../plots/mag_time_tracks_comp2.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'mag_time_tracks_nad' for task in args.plots):

    datbase_comp = ['nihao_ell_bh','nihao_nadine','nihao_nadine']
    gals_comp = ['g1.55e12','g2.37e12','g2.71e12','g6.86e12']
    times = [['0.92','0.33','0.51','1.49'],['0.70','2.12','0.75','NA'],['0.41','1.32','2.31','0.3-0.8']]
    mode = ['bondi','alpha','torque']

    def vn(arr):
        if (len(arr)==1): return arr[0]
        else: return np.nan

    # get times
    nad_ts = [[],[],[],[]]
    nad_ts[0] = ['g1.55e12',[],[]]
    nad_ts[1] = ['g2.37e12',[],[]]
    nad_ts[2] = ['g2.71e12',[],[]]
    nad_ts[3] = ['g6.86e12',[],[]]
 
    for i, gname in enumerate(['g1.55e12','g2.37e12','g2.71e12','g6.86e12']):
        nad_ts[i][0] = gname
        nad_ts[i][1] = [vn(data_all['nihao_ell_bh'][0][gname]['tg_true']), vn(data_all['nihao_nadine'][0][gname+'_alpha_1kpc_fb']['tg_true']), vn(data_all['nihao_nadine'][0][gname+'_torque_1kpc_fb']['tg_true'])]
        nad_ts[i][2] = [vn(data_all['nihao_ell_bh'][0][gname]['dt_dec']), vn(data_all['nihao_nadine'][0][gname+'_alpha_1kpc_fb']['dt_dec']), vn(data_all['nihao_nadine'][0][gname+'_torque_1kpc_fb']['dt_dec'])]
    # end get times
 

    plt.clf()
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12,8))

    for i in range(3):
        for j in range(4):
            datbase = datbase_comp[i]
            galname = gals_comp[j]
            if (i==1): galname+='_alpha_1kpc_fb'
            if (i==2): galname+='_torque_1kpc_fb'
            ax = axs[i][j]

            N = data_all[datbase][0][galname]['n_steps']
            time = [data_all[datbase][k][galname]['time'] for k in range(N)]
            mstar = [np.log10(data_all[datbase][k][galname]['mstar']) for k in range(N)]
            mbh = [data_all[datbase][k][galname]['mbh_c'] for k in range(N)]
            color = [data_all[datbase][k][galname][prfclr+'u-r'] for k in range(N)]
            sfr = [np.log10(data_all[datbase][k][galname]['sfr_ts']) for k in range(N)]

#            mdot = [1e-9*(mbh[k]-mbh[k-1])/(time[k]-time[k-1]) for k in range(1,N)]
#            for k in range(len(mdot)):
#                if (mdot[k]==0): mdot[k]=1e-5
#            mdot = np.log10(mdot)
#            time2 = [0.5*(time[k]+time[k-1]) for k in range(1,N)]

            ax.plot(time,color,'-o',ms=1,lw=1,color=colors[4],label='U-R')
            ax.plot(time,sfr,'-o',ms=1,lw=1,color=colors[1],label='SFR')
#            ax.plot(time2,mdot,'-o',ms=2,color=colors[2],label='$\dot{M}_{\mathrm{BH}}$')

            var = np.vstack((mstar,time))
            ax.plot(time,fct_mag_cut(var,*data_all['postp']['pb_time']),color='b')
            ax.plot(time,fct_mag_cut(var,*data_all['postp']['pr_time']),color='r')

            ax.set_xlim([-0.05,13.8])
            ax.set_ylim([-4.5,3.1])

            if (i!=2): ax.set_xticks([])
            if (j!=0): ax.set_yticks([])
            ax.text(0.03,0.03,galname[0:8],transform=ax.transAxes,fontsize='large')
            ax.text(0.40,0.03,mode[i],transform=ax.transAxes,fontsize='large')
#            ax.text(0.67,0.03,times[i][j]+' Gyr',transform=ax.transAxes,fontsize='large')
            ax.text(0.67,0.03,format(nad_ts[j][1][i],".2f")+' Gyr',transform=ax.transAxes,fontsize='large')

    axs[1][0].legend(loc='center right',fontsize='small')
    axs[2][1].set_xlabel('time [Gyr]')            
    axs[1][0].set_ylabel('U-R  log SFR  [M$_\odot$ yr$^{-1}$]')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/mag_time_tracks_nad.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(plot == 'lfile_tgreen' for plot in args.plots):

    plt.clf()

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(2*w_halfpage,0.66*w_halfpage))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]


    ax1.plot(data_all['lfile']['dts'],data_all['lfile']['tgreen_dt'], color=colors[0],label='dt')
    ax2.plot(data_all['lfile']['tqs'],data_all['lfile']['tgreen_tq'], color=colors[1],label='tq')
    ax3.plot(1000*np.array(data_all['lfile']['mets']),data_all['lfile']['tgreen_met'], color=colors[2],label='met')


    ax1.set_ylabel('$\\tau_{\mathrm{SSP}}$')
    ax1.set_xlabel('$T_{\mathrm{SFQ}}$')
    ax2.set_xlabel('$t_{\mathrm{q}}$')
    ax3.set_xlabel('Z x 1000')

    ax1.set_xlim([0.0,3.3])
    ax2.set_xlim([-0.1,10.5])
    ax3.set_xlim([8,42])

    ax1.set_ylim([0,13])
    ax2.set_ylim([0,13])
    ax3.set_ylim([0,13])

    ax2.set_yticks([])
    ax3.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/lfile_tgreen.'+args.pictype,dpi=1000,bbox_inches='tight')
 




if any(task == 'lfile_sfr' for task in args.plots):

    plt.clf()

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(w_halfpage,0.75*w_halfpage))

    time = np.arange(0.1,14,0.2)
    var = np.vstack((np.log10(1e10*time),time))
    mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
    mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])


    axs.plot(data_all['lfile']['time'],data_all['lfile']['sfr_const'],'-',color=colors[5],label='const')
    axs.plot(data_all['lfile']['time'],data_all['lfile']['sfr_exp'],'-',color=colors[1],label='$\exp(-t\,/\,1\,\mathrm{Gyr})$')
    axs.plot(data_all['lfile']['time'],data_all['lfile']['sfr_t-2'],'-',color=colors[2],label='$(t\,/\,1\,\mathrm{Gyr})^{-2}$')
    axs.plot(data_all['lfile']['time'],data_all['lfile']['sfr_SB'],'-',color=colors[6],label='SB at 4 Gyr')
    axs.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq'],'-',color=colors[4],label='$t_{\mathrm{q}}$=5 Gyr')

    axs.plot(time,mag_cut_b,c='b')
    axs.plot(time,mag_cut_r,c='r')

    axs.legend(loc='lower right',fontsize='small')

    axs.set_xlim([0,14])
    axs.set_ylim([0.5,3.2])
    axs.set_ylabel('U-R')
    axs.set_xlabel('time [Gyr]')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/lfile_sfr.'+args.pictype,dpi=1000,bbox_inches='tight')






if any(task == 'lfile_sfr_old' for task in args.plots):

    plt.clf()

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(w_halfpage,2.5*w_halfpage))

    time = np.arange(0.1,14,0.2)
    var = np.vstack((np.log10(1e10*time),time))
    mag_cut_b = fct_mag_cut(var,*data_all['postp']['pb_time'])
    mag_cut_r = fct_mag_cut(var,*data_all['postp']['pr_time'])


    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]
    
#####ax1

    ax1.plot(data_all['lfile']['time'],data_all['lfile']['sfr_const'],'-',color=colors[0],label='const')
    ax1.plot(data_all['lfile']['time'],data_all['lfile']['sfr_exp'],'-',color=colors[1],label='$\exp(-t\,/\,1\,\mathrm{Gyr})$')
    ax1.plot(data_all['lfile']['time'],data_all['lfile']['sfr_t-2'],'-',color=colors[2],label='$(t\,/\,1\,\mathrm{Gyr})^{-2}$')
    ax1.plot(data_all['lfile']['time'],data_all['lfile']['sfr_SB'],'-',color=colors[3],label='SB at 4 Gyr')
    ax1.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq'],'-',color=colors[4],label='$t_{\mathrm{q}}$=5 Gyr')

    ax1.plot(time,mag_cut_b,c='b')
    ax1.plot(time,mag_cut_r,c='r')

    ax1.legend(loc='lower right',fontsize='small')


#####ax2

    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_dt2'],'-',color=colors[2],label='$T_{\mathrm{SFQ}}$=2 Gyr')
    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_dt1'],'-',color=colors[1],label='$T_{\mathrm{SFQ}}$=1 Gyr')
    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_dt05'],'-',color=colors[3],label='$T_{\mathrm{SFQ}}$=0.5 Gyr')
    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_dt02'],'-',color=colors[5],label='$T_{\mathrm{SFQ}}$=0.2 Gyr')
#    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_dt01'],'-',color=colors[4],label='$T_{\mathrm{SFQ}}$=0.1 Gyr')
    ax2.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq'],'-',color=colors[4],label='$T_{\mathrm{SFQ}}$=0 Gyr')

    ax2.plot(time,mag_cut_b,c='b')
    ax2.plot(time,mag_cut_r,c='r')

    ax2.legend(loc='lower right',fontsize='small')

#####ax3

    ax3.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq1'],'-',color=colors[2],label='$t_{\mathrm{q}}$=1 Gyr')
    ax3.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq3'],'-',color=colors[3],label='$t_{\mathrm{q}}$=3 Gyr')
    ax3.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq5'],'-',color=colors[1],label='$t_{\mathrm{q}}$=5 Gyr')
    ax3.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq10'],'-',color=colors[5],label='$t_{\mathrm{q}}$=10 Gyr')
    ax3.plot(data_all['lfile']['time'],data_all['lfile']['sfr_tq20'],'-',color=colors[0],label='NA')

    ax3.plot(time,mag_cut_b,c='b')
    ax3.plot(time,mag_cut_r,c='r')

    ax3.legend(loc='lower right',fontsize='small')

#####

    ax4.plot(data_all['lfile']['time'],data_all['lfile']['sfr_met1'],'-',color=colors[2],label='$Z=0.01$')
    ax4.plot(data_all['lfile']['time'],data_all['lfile']['sfr_met2'],'-',color=colors[1],label='$Z=0.02$')
    ax4.plot(data_all['lfile']['time'],data_all['lfile']['sfr_met3'],'-',color=colors[3],label='$Z=0.03$')
    ax4.plot(data_all['lfile']['time'],data_all['lfile']['sfr_met4'],'-',color=colors[5],label='$Z=0.04$')

    ax4.plot(time,mag_cut_b,c='b')
    ax4.plot(time,mag_cut_r,c='r')

    ax4.legend(loc='lower right',fontsize='small')

####

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_xlim([0,14])
    ax2.set_xlim([0,14])
    ax3.set_xlim([0,14])
    ax4.set_xlim([0,14])

    ax1.set_ylim([0.5,3.2])
    ax2.set_ylim([0.5,3.2])
    ax3.set_ylim([0.5,3.2])
    ax4.set_ylim([0.5,3.2])

    ax2.set_ylabel('U-R')
    ax4.set_xlabel('time [Gyr]')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/lfile_sfr_old.'+args.pictype,dpi=1000,bbox_inches='tight')






if any(task == 'time_mstar' for task in args.plots):

    nts = 64
    ts_Gyr = 0.215769412

    tr = nts*ts_Gyr - np.array(data_all['postp']['tr_true'])
    tg = data_all['postp']['tg_true']
    tb = data_all['postp']['tb_true']
    mstar_z0 = np.log10(data_all['postp']['mstar_z0'])

    N=5
    mstar_av = np.convolve(sorted(mstar_z0), np.ones(N)/N, mode='valid')
    tr_av = np.convolve([x for _,x in sorted(zip(mstar_z0,tr))], np.ones(N)/N, mode='valid')
    tg_av = np.convolve([x for _,x in sorted(zip(mstar_z0,tg))], np.ones(N)/N, mode='valid')
    tb_av = np.convolve([x for _,x in sorted(zip(mstar_z0,tb))], np.ones(N)/N, mode='valid')

#    print(np.median(tr),np.median(tg),np.median(tb))


    plt.clf()
    fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

    ax.scatter(mstar_z0,tr,marker='o',edgecolors='none',color='r')
    ax.scatter(mstar_z0,tg,marker='s',edgecolors='none',color='g')
    ax.scatter(mstar_z0,tb,marker='v',edgecolors='none',color='b')

    ax.plot(mstar_av,tr_av,'-',color='r')
    ax.plot(mstar_av,tg_av,'-',color='g')
    ax.plot(mstar_av,tb_av,'-',color='b')

    ax.set_xlabel('log $M_{\star}$ [M$_\odot$]')
    ax.set_ylabel('time [Gyr]')
#    ax.legend(loc='lower right',fontsize='xx-small')
    ax.set_ylim([-0.1,14])

    plt.savefig('../plots/time_mstar.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(task == 'timescales' for task in args.plots):

    print('timescales')

    plt.clf() 
    fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

    dt_dec = np.array(data_all['postp']['dt_dec'])
    sfr_mon = np.array(data_all['postp']['sfr_mon'])
    tgreen = np.array(data_all['postp']['tg_true'])

    flt1 = np.where(sfr_mon)[0]
    ax.scatter(dt_dec[flt1],tgreen[flt1],marker='o',s=10,color=colors[0])
    popt, pcov = scipy.optimize.curve_fit(fct_lin,dt_dec[flt1],tgreen[flt1],p0=[2.,0.])
    ax.plot(dt_dec[flt1],fct_lin(dt_dec[flt1],*popt),color=colors[0],label='$\\tau_{\mathrm{SFON}}$')
    print('    sims: ',popt,np.sqrt(np.diag(pcov)))


    dt_dec = data_all['lfile']['dts'][0:3]
    tgreen = data_all['lfile']['tgreen_dt'][0:3]
    popt, pcov = scipy.optimize.curve_fit(fct_lin,dt_dec,tgreen,p0=[2.,0.])
    ax.plot(dt_dec,tgreen, color=colors[1],label='$\\tau_{\mathrm{SSP}}$')
    print('    ssps: ',popt,np.sqrt(np.diag(pcov)))

    ax.text(0.2,0.75,'$\\frac{\\tau}{T_{\mathrm{SFQ}}} \\approx 2$',transform=ax.transAxes,fontsize='xx-large')

    ax.set_ylabel('$\\tau$ [Gyr]')
    ax.set_xlabel('$T_{\mathrm{SFQ}}$ [Gyr]')

    ax.set_xlim([0,1])

    ax.legend(loc='lower right')

    plt.savefig('../plots/timescales.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(plot == 'times_nad' for plot in args.plots):

    nad_ts = [[],[],[]]
    nad_ts[0] = ['g1.55e12',[956,694,878],[566,390,457]]
    nad_ts[1] = ['g2.37e12',[1277,2614,309],[613,1221,157]]
    nad_ts[2] = ['g2.71e12',[2317,781,1206],[1535,594,186]]

    nad_ts[0] = ['g1.55e12',[0.88,0.69,0.96],[0.46,0.39,0.57]]
    nad_ts[1] = ['g2.37e12',[0.31,2.61,1.28],[0.16,1.22,0.61]]
    nad_ts[2] = ['g2.71e12',[1.21,0.78,2.32],[0.19,0.59,1.54]]


    x = ['bondi','alpha','torque']
    lss = ['dashed','dotted','dashdot']
    plt.clf()
    fig, ax = plt.subplots(figsize=(w_halfpage,0.75*w_halfpage))

    for i in range(len(nad_ts)):
        ax.plot(x,nad_ts[i][1],c=colors[7],ls=lss[i],label=nad_ts[i][0])
        ax.plot(x,nad_ts[i][1],c=colors[0],ls=lss[i])
        ax.plot(x,nad_ts[i][2],c=colors[1],ls=lss[i])

    ax.legend()
    ax.set_ylabel('$\\tau$, $T_{\mathrm{SFQ}}$ [Gyr]')
    plt.savefig('../plots/times_nad.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(plot == 'hist_times' for plot in args.plots):

    tg_true = np.array(data_all['postp']['tg_true'])
    dt_simp = np.array(data_all['postp']['tg_simp'])
    dt_sfon = np.array(data_all['postp']['tg_sfon'])-dt_simp
    dt_merg = tg_true-np.array(data_all['postp']['tg_sfon'])
    dt_dec = np.array(data_all['postp']['dt_dec'])
    
    print(dt_dec)

    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(2*w_halfpage,0.66*w_halfpage))

    axs[0].hist(dt_simp,bins=15)
    axs[1].hist(dt_sfon,bins=15)
    axs[2].hist(dt_merg,bins=15)

    axs[0].axvline(x=np.median(dt_simp),color='k')
    axs[1].axvline(x=np.median(dt_sfon[np.where(~np.isnan(dt_sfon))[0]]),color='k')
    axs[2].axvline(x=np.median(np.abs(dt_merg[np.where(~np.isnan(dt_merg))[0]])),color='k')
    print('hist_times')
    print("    mean dt_dec:  %.4f " %np.mean(dt_dec[np.where(dt_dec>0)[0]]))
    print("    mean tg_true:  %.4f " %np.mean(tg_true))
    print("    times means: simp = %.4f, sfon = %.4f, xs = %.4f" %(np.mean(dt_simp),np.mean(dt_sfon[np.where(~np.isnan(dt_sfon))[0]]),np.mean(np.abs(dt_merg[np.where(~np.isnan(dt_merg))[0]]))))

    axs[0].set_xlabel('$\\tau_{\mathrm{simp}}$')
    axs[1].set_xlabel('$\\Delta \\tau_{\mathrm{SFON}}$')
    axs[2].set_xlabel('$\\Delta \\tau_{\mathrm{XS}}$')
    axs[0].set_ylabel('number')

#    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/hist_times.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(plot == 'tg_simp' for plot in args.plots):
    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(2*w_halfpage,0.66*w_halfpage))

    x1 = np.log10(data_all['postp']['mstar_tb'])
    x2 = data_all['postp']['star_age_mean_tb']
    x3 = data_all['postp']['star_metals_mean_tb']
    y = data_all['postp']['tg_simp']

    print('tg_simp, mean tg_simp = %.2f' %np.mean(np.array(y)))
 
    axs[0].scatter(x1,y,s=5,color=colors[0],label='$\\tau_{\mathrm{SFON}}$')
    axs[1].scatter(x2,y,s=5,color=colors[0],label='$\\tau_{\mathrm{SFON}}$')
    axs[2].scatter(x3,y,s=5,color=colors[0],label='$\\tau_{\mathrm{SFON}}$')

    axs[0].plot(data_all['postp']['lfile_simp_mstar'],data_all['postp']['lfile_simp_tg_mstar'],color=colors[1],label='$\\tau_{\mathrm{SSP}}$')
    axs[1].plot(data_all['postp']['lfile_simp_ages'][:-1],data_all['postp']['lfile_simp_tg_ages'][:-1],color=colors[1],label='$\\tau_{\mathrm{SSP}}$')
    axs[2].plot(data_all['postp']['lfile_simp_mets'][1:],data_all['postp']['lfile_simp_tg_mets'][1:],color=colors[1],label='$\\tau_{\mathrm{SSP}}$')

    axs[0].set_xlabel('log $M_{\star,\mathrm{b}}$ [M$_\odot$]')
    axs[1].set_xlabel('age at $t_{\mathrm{b}}$ [Gyr]')
    axs[2].set_xlabel('$Z_{\mathrm{b}}$')
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].set_ylabel('$\\tau$ [Gyr]')
    axs[2].set_xlim([0.001,0.042])
    axs[0].set_ylim([0,1.3])
    axs[1].set_ylim([0,1.3])
    axs[2].set_ylim([0,1.3])

    axs[1].legend(loc='upper left')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/tg_simp.'+args.pictype,dpi=1000,bbox_inches='tight')





if any(plot == 'tg_sfon' for plot in args.plots):
    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*w_halfpage,0.66*w_halfpage))

    x1 = np.log10(data_all['postp']['mass_sfonly'])
    x2 = data_all['postp']['age_sfonly']
    y = np.array(data_all['postp']['tg_sfon']) # - np.array(data_all['postp']['tg_simp'])

    res = scipy.stats.spearmanr(x1,y)
    print('tg_sfon spearman corr:   %.2f %.2e' %res)

    axs[0].scatter(x1,y,s=5,color=colors[0],label='$\\tau_{\mathrm{SFON}}$')
    axs[1].scatter(x2,y,s=5,color=colors[0],label='$\\tau_{\mathrm{SFON}}$')

    y = data_all['lfile']['sfon_tg']
    x1 = np.log10(data_all['lfile']['sfon_dM'])
    x2 = data_all['lfile']['sfon_age']

    axs[0].plot(x1[:-1],y[:-1],color=colors[1],label='$\\tau_{\mathrm{SSP}}$')
    axs[1].plot(x2[:-1],y[:-1],color=colors[1],label='$\\tau_{\mathrm{SSP}}$')

    axs[0].set_xlabel('log $\Delta M_{\star}$ [M$_\odot$]')
    axs[1].set_xlabel('age [Gyr]')
    axs[1].set_yticks([])
    axs[0].set_ylabel('$\\tau$ [Gyr]')

    axs[1].legend(loc='upper left')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../plots/tg_sfon.'+args.pictype,dpi=1000,bbox_inches='tight')


