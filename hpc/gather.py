
import argparse
import json
import multiprocessing
import numpy as np
import os

from functions import *



###
##### options parsing
###

parser = argparse.ArgumentParser(description='calc and plot')
parser.add_argument('-t', '--threads', type=int, help='number of threads',default=1)
parser.add_argument('-c', '--cores', type=int, help='number of cores',default=16)
parser.add_argument('-k', '--tasks', nargs='+', help='what to do')
args = parser.parse_args()

pynbody.config['number_of_threads'] = 1
os.environ['OMP_NUM_THREADS']=str(args.threads)





###
##### Set simulations and files
###

# elliptical galaxies with BH, resolution level A
ells1 = ['g1.26e12','g1.27e12','g1.55e12','g1.62e12','g2.37e12','g2.71e12','g3.74e12'] 

# elliptical galaxies with BH, resolution level A
ells2 = ['g4.41e12','g4.55e12','g4.81e12','g4.84e12','g5.22e12','g5.41e12','g5.53e12','g6.53e12','g6.70e12','g7.50e12','g7.55e12','g7.71e12','g7.92e12','g8.08e12','g8.45e12','g8.94e12','g9.61e12','g1.05e13','g1.14e13','g1.17e13','g1.25e13','g1.33e13','g1.44e13','g1.54e13','g1.57e13','g1.63e13','g1.87e13','g2.02e13','g2.07e13','g2.11e13','g2.20e13','g2.37e13','g3.26e13','g6.57e12','g6.86e12','g3.42e12','g2.58e13','g3.78e13','g3.89e13','g2.10e13']

# classic nihao galaxies, whose DMO counterpart is not polluted or off-mass
nihao_classic_ydm = ['g1.23e10', 'g1.52e11', 'g2.34e10', 'g3.19e10', 'g3.54e09', 'g3.71e11', 'g5.02e11', 'g5.38e11', 'g1.08e11', 'g1.37e11', 'g1.57e10', 'g1.95e10', 'g2.37e10', 'g2.80e10', 'g3.21e11', 'g3.55e11', 'g5.05e10', 'g5.41e09', 'g8.26e11', 'g1.44e10', 'g1.57e11', 'g1.89e10', 'g2.39e10', 'g2.83e10', 'g3.59e11', 'g4.94e10', 'g5.22e09', 'g7.05e09', 'g7.55e11', 'g1.47e10', 'g1.59e11', 'g2.09e10', 'g3.44e10', 'g4.36e09', 'g4.99e09', 'g8.89e10', 'g1.18e10', 'g1.50e10', 'g1.64e11', 'g1.92e10', 'g2.19e11', 'g2.64e10', 'g3.67e10', 'g4.48e10', 'g5.59e09', 'g6.91e10', 'g8.06e11', 'g9.26e09', 'g2.39e11']

# classic nihao galaxies, whose DMO counterpart is polluted, off-mass or non-existent
nihao_classic_ndm = ['g1.05e11', 'g1.92e12', 'g2.42e11', 'g2.79e12', 'g4.86e10', 'g5.84e09', 'g6.96e10', 'g9.59e10', 'g1.88e10', 'g2.54e11', 'g3.93e10', 'g4.90e11', 'g6.96e11', 'g3.23e11', 'g5.46e11', 'g6.37e10', 'g1.12e12', 'g2.63e10', 'g5.55e11', 'g6.77e10', 'g2.41e11', 'g3.06e11', 'g4.99e10', 'g2.57e11', 'g3.49e11', 'g3.61e11', 'g6.12e10', 'g3.67e09', 'g4.48e09', 'g8.63e09', 'g9.91e09']


galaxies = dict()
# classic nihao galaxies with BH
galaxies['nihao_bh'] = ['g7.55e11','g8.26e11','g1.12e12','g1.92e12','g2.79e12']
# elliptical galaxies without BH
galaxies['nihao_ell_wobh'] = ['g1.05e13','g1.44e13','g5.41e12','g6.86e12','g7.50e12','g7.92e12']
# all elliptical galaxies with BH
galaxies['nihao_ell_bh'] = ells1+ells2
# all classic nihao galaxies
galaxies['nihao_classic'] = nihao_classic_ydm+nihao_classic_ndm
galaxies['nihao_nadine'] = ['g1.55e12_alpha_1kpc_fb','g2.37e12_alpha_1kpc_fb','g2.71e12_alpha_1kpc_fb','g6.86e12_alpha_1kpc_fb','g1.55e12_torque_1kpc_fb','g2.37e12_torque_1kpc_fb','g2.71e12_torque_1kpc_fb','g6.86e12_torque_1kpc_fb']


base = '/data/database/nihao/'
databases = ['nihao_classic','nihao_bh','nihao_ell_bh','nihao_ell_wobh','nihao_nadine']
###
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





####################
####################





###
##### Load all data
###

paramarray = []

for datbase in databases:

    for galname in galaxies[datbase]:
        filename = directories[datbase]+'/'+galname+'/haloids.json'
        if os.path.exists(filename):
            f2 = open(filename, 'r')
            haloids = json.load(f2)
            f2.close()
        else: continue

        N = len(np.where(np.array(haloids['haloids'][0])>=0)[0])
        if (N==0): continue

        for i in range(N):
            prms = [datbase,i,galname]
            paramarray.append(prms)
print(len(paramarray))




###
##### Calculate individual galaxy data
###

def startsnap(prms):
    os.system('python calc_idv.py -d '+prms[0]+' -i '+str(prms[1])+' -g '+prms[2])
    return 0


if any(task == 'calc_idv' for task in args.tasks):
    print('start pooling')
    pool = multiprocessing.Pool(args.cores)
    results = pool.map(startsnap, paramarray)
    pool.close()
    print('end pooling')





###
##### Gather individual galaxy data
###

if any(task == 'gather' for task in args.tasks):

    data_all = dict()
    data_all['galaxies'] = galaxies
    data_all['databases'] = databases
    data_all['directories'] = directories

    for datbase in databases:
        data_all[datbase] = []
        for i in range(nts): data_all[datbase].append(dict())

    for prms in paramarray:
        datbase = prms[0]; i = prms[1]; galname = prms[2]
        cmd = 'python calc_idv.py -d '+prms[0]+' -i '+str(prms[1])+' -g '+prms[2]+' &'
        filename = 'data/'+prms[0]+'_'+prms[2]+'_'+str(prms[1])+'.json'
        if os.path.exists(filename):
            f=open(filename,'r')
            try:
                tempdct = json.load(f)
            except:
                print('tempdct for', filename, 'could not be loaded')
                os.system(cmd)
            f.close()
        else:
            print('file', filename, 'does not exist')
            os.system(cmd)

        data_all[datbase][i][galname] = tempdct

    galaxies = data_all['galaxies']
    databases = data_all['databases']

    f=open('data_all.json','w')
    f.write(json.dumps(data_all))
    f.close()


