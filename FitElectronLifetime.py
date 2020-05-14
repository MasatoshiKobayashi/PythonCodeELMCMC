import numpy as np

import time
import copy
import click
import pickle
import emcee
import argparse

import ElectronLifetimeTrend
from ElectronLifetimeTrend import *

import FormPars 
from FormPars import *

import LnLike 
from LnLike import *

import MCMC_Tools
from MCMC_Tools import *

import Fit_func2


StartingTimeFit = time.time()

SSELifeDataFile ='/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeData_SingleScatter_ForV2Fitting.txt'
PoRnELifeDataFile =  '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithPoRn_mod.txt'
Po218ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithPo218_1day_new_wobad.txt'
Rn220ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn220_1day_new_wobad.txt'
Rn222ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222_1day_new_UB.txt'
FileParams = 'TEMP_Param.txt'

FormPars = MyFormPars()
MinUnixTime  = GetUnixTimeFromTimeStamp(FormPars.GetMinTimeStamp())
MaxUnixTime  = GetUnixTimeFromTimeStamp(FormPars.GetMaxTimeStamp())

parser = argparse.ArgumentParser()
parser.add_argument('historian_file', type=str, help='Path to historian pickle file')
parser.add_argument('output_file', type=str, help='Output file name')
parser.add_argument('nwalkers', type=int, help='Number of walkers')
parser.add_argument('niterations', type=int, help='Number of iterations')
parser.add_argument('--nthreads', type=int, help='Number of threads', default=1)
parser.add_argument('--pre_fit_file', type=str, help='Posterior pickle file from previous fit')
parser.add_argument('--include_po_rn', help='Include Po-Rn combined fits', default=True)
parser.add_argument('--ss_file', type=str, help='Single scatter electron lifetimes', default=SSELifeDataFile)
parser.add_argument('--po_rn_file', type=str, help='Po-Rn combined fit lifetimes', default=PoRnELifeDataFile)
parser.add_argument('--rn220_file', type=str, help='Rn-220 electron lifetimes', default=Rn220ELifeDataFile)
parser.add_argument('--rn222_file', type=str, help='Rn-222 electron lifetimes', default=Rn222ELifeDataFile)
parser.add_argument('--po218_file', type=str, help='Po-218 electron lifetimes', default=Po218ELifeDataFile)
parser.add_argument('--min_unixtime', type=int, help='Minimum unixtime', default=MinUnixTime)
parser.add_argument('--max_unixtime', type=int, help='Maximum unixtime', default=MaxUnixTime)
parser.add_argument('--verbose', '-v', type=int, choices=[0, 1, 2, 3], help='Verbose (0-3)', default=1)
parser.add_argument('--include_firstss',type=int, choices=[0, 1], help='Include first ss combined fits', default=1)

args = parser.parse_args()

HistorianFile                = args.historian_file
FitOutput                    = args.output_file
nwalkers                     = args.nwalkers
niterations                  = args.niterations
nthreads                     = args.nthreads
PreWalkingPickleFilename     = args.pre_fit_file
IncludePoRn                  = args.include_po_rn
IncludeFirstSS               = args.include_firstss
SSELifeDataFile              = args.ss_file
PoRnELifeDataFile            = args.po_rn_file
Po218ELifeDataFile           = args.po218_file
Rn220ELifeDataFile           = args.rn220_file
Rn222ELifeDataFile           = args.rn222_file
MinUnixTime                  = args.min_unixtime
MaxUnixTime                  = args.max_unixtime
verbose                      = args.verbose

print('\nFitting Electron Lifetime between ' + str(MinUnixTime) + ' and ' + str(MaxUnixTime) + '\n')

print('\nRunning with %i threads\n' %nthreads)
PredictionOutputFile=''
ConditionSets = (nwalkers,niterations,nthreads,IncludePoRn,MinUnixTime,MaxUnixTime,verbose,IncludeFirstSS)
FileSets = (HistorianFile,FitOutput,PredictionOutputFile,FileParams)
DataSets = ( SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
FormPars = MyFormPars()
Fit_func = Fit_func2.MyFit_func()
Fit_func.Init(ConditionSets,FileSets,DataSets)
print("Done!")
UsePreWalking=0
if PreWalkingPickleFilename is not None:
    UsePreWalking=1

####################
# start MCMC fitting
####################
# initial parameters
x0, x0_steps = FormPars.GetOldBestParameters()
print("Initial Params")
print(x0)
tmp,tmpcheck,parindex=FormPars.FormPars(x0)
print(tmp)

ndim = len(x0)

#if not PreWalkingPickleFilename=="NoneExist":
#if PreWalkingPickleFilename is not None:
chi2_thre = 1e10
if UsePreWalking:
    print('Using pre-walker pickle for fit')
    PreWalkingData = pickle.load(open(PreWalkingPickleFilename, 'rb'))
    chain = PreWalkingData['sampler']._chain
    chi2 = PreWalkingData['sampler'].__dict__['_lnprob']
    pre_walkers = PreWalkingData['sampler'].__dict__['k']
    pre_iterations = PreWalkingData['sampler'].__dict__['iterations']
    p_pre = chain[:, pre_iterations-1, :]
    p0 = []
    nwalkers=0
    for i in range(pre_walkers):
        chi2_test = chi2[i,  pre_iterations-1 ]*(-2)
        if chi2_test>chi2_thre:
            print('TOO bad chi2. Skip this walker: ',chi2_test)
        else:
            p0.append(p_pre[i])
            nwalkers += 1
    if nwalkers%2 == 1:
        print('Reduce one walker to make it even ')
        p0 = p0[:-1]
        nwalkers -= 1

else:
    # randomize the walkers
    print(nwalkers)
    p0 = []
    for i in range(nwalkers):
        lo_bound, hi_bound = FormPars.GetInitialBounds(x0,x0_steps)
        rand=[]
        for j in range(len(lo_bound)) :
            rand.append(np.random.uniform(lo_bound[j],hi_bound[j]))
        tmp,tmpcheck,parindex=FormPars.FormPars(rand)
        if tmpcheck:
            raise ValueError('Out Of boundary for initial parameters ',parindex,np.asarray(rand))
        p0.append(rand)
print("Use " + str(nwalkers)+" walkers")

# emcee explained in https://arxiv.org/abs/1202.3665
#sampler = emcee.EnsembleSampler(nwalkers, ndim, LnLike, threads=nthreads)
# from Matt's github https://github.com/mdanthony17/emcee
# Differential Evolution Markov Chain (DEMC) with snooker updates
# https://link.springer.com/article/10.1007/s11222-006-8769-1
# https://link.springer.com/article/10.1007/s11222-008-9104-9
sampler = emcee.DESampler(nwalkers, ndim, LnLike, threads=nthreads, live_dangerously=True)

with click.progressbar(sampler.sample(p0=p0, iterations=niterations), length=niterations) as mcmc_sampler:
    for i,results in enumerate(mcmc_sampler):
        # save every 500 iterations
        if (i+1)%500 == 0:
            if verbose >= 1:
                print('making sampler', time.time())
            temp_sampler = copy.copy(sampler)
            del temp_sampler.__dict__['lnprobfn']
            del temp_sampler.__dict__['pool']

            OutputData = {}
            OutputData['prefilename']=PreWalkingPickleFilename
            OutputData['sampler'] = temp_sampler
            pickle.dump(OutputData, open(FitOutput, 'wb'))

            del temp_sampler
        pass

#sampler = Fit_func.DoMCMC()

# delete un-pickleable parts of sampler
del sampler.__dict__['lnprobfn']
del sampler.__dict__['pool']

OutputData = {}
OutputData['prefilename']=PreWalkingPickleFilename
OutputData['sampler'] = sampler
OutputData['args'] = args.__dict__

# save electron lifetimes used in fit
ElectronLifetimeData =  Fit_func.GetElectronLifetimeData()
for key in ElectronLifetimeData.keys():
    OutputData['ElectronLifetimeData' + key] = ElectronLifetimeData[key]

pickle.dump(OutputData, open(FitOutput, 'wb'),protocol=4)

print('Fit took {0} hours'.format(str((time.time() - StartingTimeFit)/3600.)))
