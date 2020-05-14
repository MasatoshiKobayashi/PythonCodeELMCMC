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

import MCMC_Tools
from MCMC_Tools import *

import Fit_func2

from iminuit import Minuit

StartingTimeFit = time.time()
NumOfInterpolation = 1000

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
parser.add_argument('prediction_output_file', type=str, help='Path to prediction output file')
parser.add_argument('--output_file', type=str, help='Output file name', default='')
parser.add_argument('--nwalkers', type=int, help='Number of walkers', default=0)
parser.add_argument('--niterations', type=int, help='Number of iterations', default=0)
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
PredictionOutputFile         = args.prediction_output_file
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
ModeMinuit=True
ConditionSets = (nwalkers,niterations,nthreads,IncludePoRn,MinUnixTime,MaxUnixTime,verbose,IncludeFirstSS)
FileSets = (HistorianFile,FitOutput,PredictionOutputFile,FileParams,ModeMinuit)
DataSets = ( SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
FormPars = MyFormPars()
Fit_func = MyFit_func()
Fit_func.Init(ConditionSets,FileSets,DataSets)
print("Setting Files...")
Limits = FormPars.GetParLimits()
print("Done!")
ElectronLifetimeData =  Fit_func.GetElectronLifetimeData()
DUnixTimes=ElectronLifetimeData['UnixTimes'] 
DValues=ElectronLifetimeData['Values'] 
DValueErrors=ElectronLifetimeData['ValueErrors'] 
FitOutputTXT = FitOutput+"_Minuit"


####################
# start Minuit fitting
####################
# initial parameters
Fix=[]
x0 = MCMC_Tools.LoadParams(FileParams)
for i in range(len(x0)):
    Fix.append(False)
print("Initial Params")
print(x0)
tmp,tmpcheck,parindex=FormPars.FormPars(x0)
if tmpcheck:
    raise ValueError('Out Of boundary for initial parameters ',parindex,np.asarray(rand))

m = Minuit.from_array_func(LnLikeChi2,x0, error=np.asarray(x0)*0.1, limit=Limits, fix=Fix, errordef=1) 
m.set_strategy(1)
#fmin,param = m.migrad(precision=0.1)
fmin,param = m.migrad(ncall=15000)

Par=[]
for i in range(len(param)):
    Par.append(param[i]['value'])

fout ="Param_Result.txt"
dumpTXT(fout,Par,Par,P=True)
