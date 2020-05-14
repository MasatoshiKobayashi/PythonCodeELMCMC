import ElectronLifetimeTrend
from ElectronLifetimeTrend import *

import MCMC_Tools
from MCMC_Tools import *

import FormPars

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv

#import Fit_func2
#from  Fit_func2 import *

import matplotlib.pyplot as plt

if len(sys.argv)<2:
    print("======= Syntax ========")
    print("python PredictElectronLifetime.py")
    print("<Historian file>")
    print("<Fit result pickle>")
    print("<Prediction txt output>")
    print("<burn-in iteraction cut>")
    print("NOTICE: do remember to add \"-r\" at the end")
    exit()

HistorianFile = sys.argv[1]
FitResultInput = sys.argv[2]
PredictionOutputFile = sys.argv[3]
BurnInCutOff = int(sys.argv[4])


NumOfInterpolation = 2000
NumOfTrials = 400


# setting the parameters
MinUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMinTimeStamp())
#MaxUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMaxTimeStamp()) 
MaxUnixTime = 1533081600 
default_pars = FormPars.GetDefaultPars()


#############################
## Get the MCMC result
#############################
import pickle
MCMCResults = pickle.load(open(FitResultInput, 'rb'))
sampler = MCMCResults['sampler']
ndim = sampler.__dict__['dim']
niterations = sampler.__dict__['iterations']
nwalkers = sampler.__dict__['k']
print(sampler)
print(ndim, nwalkers, niterations)

# Cutting Burn-In and unreasonable region
# the value shall already be from "PlotElectronLifetime.py"
#samples_cut = GetBurnInCutoffSamples(sampler.__dict__['_chain'], int(BurnInCutOff))
samples_cut = GetBurnInCutoffSamplesV2(sampler.__dict__['_chain'], int(BurnInCutOff), int(BurnInCutOff)+49)
print(len(samples_cut))

#####################################
## Calculate the best fit from MCMC results
#####################################
#MaxUnixTime = LastPointUnixTime + 60.*3600.*24. # 2 month after the last data point
# The main Light yield Trend
#pElectronLifetimeTrend = MyElectronLifetimeTrend(HistorianFile, MinUnixTime, MaxUnixTime, Pars)

SSELifeDataFile ='/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeData_SingleScatter_ForV2Fitting.txt'
PoRnELifeDataFile =  '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithPoRn.txt'
Po218ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithPo218_1day_new_wobad.txt'
Rn220ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn220_1day_new_wobad.txt'
Rn222ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222_1day_new_wobad.txt'
FileParams = '/dali/lgrandi/kobayashi1/xenon/analysis/ElectronLifetime/ElectronLifetime/PythonCodeELMCMCndna/TEMP_Param.txt'

#SetGlobalPars(nwalkers,niterations,28,1,MinUnixTime,MaxUnixTime,0,1) 
#x0, x0_steps =SetFiles(HistorianFile,FitOutput,PredictionOutputFile,FileParams)
#print("Setting Files...")
#ElectronLifetimeData=SetELifeData(SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
#ElectronLifetimeData=SetELifeDataDiv(SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
#print("Done!")


#DUnixTimes=ElectronLifetimeData['UnixTimes'] 
#DValues=ElectronLifetimeData['Values'] 
#DValueErrors=ElectronLifetimeData['ValueErrors'] 


## get the graphs 
#UnixTimes = np.linspace(MinUnixTime, MaxUnixTime, NumOfInterpolation)
## include times of discontinuity
#ImpactfulUnixtimes = FormPars.GetImpactfulUnixtimes()
#for ImpactfulUnixtime in ImpactfulUnixtimes:
#	if ImpactfulUnixtime-1 not in UnixTimes:
#		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime-1]), ImpactfulUnixtime-1)
#	if ImpactfulUnixtime not in UnixTimes:
#		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime]), ImpactfulUnixtime)
#	if ImpactfulUnixtime+1 not in UnixTimes:
#		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime+1]), ImpactfulUnixtime+1)

#############################
# calculate:
# 1) Maximum
# 2) Reaching date for 90% capacity
# 3) Reaching date for 500us
#############################
# The electron lifetime Trend for varied parameters

# function for checking the list percentile with np.inf in it
def PercentileWithInf(Values, deviation):
    TotalCounter = len(Values)
    InfCounter = 0
    NewValues = []
    # first get the new list and counter
    for value in Values:
        if value==np.inf:
            InfCounter += 1
        else:
            NewValues.append(value)
    Fraction = (1. - float(InfCounter) / float(TotalCounter)) * 100.
    if deviation>Fraction:
        return np.inf
    deviation_new = deviation / Fraction *100.
    return np.percentile(NewValues, deviation_new, axis=0)

#Print Parameters
mean = np.average(samples_cut, axis=0)
samples_cut = np.asarray(samples_cut)
print(samples_cut[0])
print(samples_cut[1])
print(samples_cut[2])
c=np.c_[samples_cut[0],samples_cut[1]]
print(len(samples_cut[0]))
print(c)
tot=samples_cut[0]
for i in range(1,len(samples_cut)):
    if(i%1000==0):
        print("Now " +str(i))
    tot=np.c_[tot,samples_cut[i]]

StandardDeviation1Sigma = [15.4, 50., 84.6]
print(mean)

Lower=[]
Center=[]
Upper=[]
for i, deviation in enumerate(StandardDeviation1Sigma):
    for Values in tot:
        Boundary = PercentileWithInf(Values, deviation)
        if i==0:
            Lower.append(Boundary)
        elif i==1:
            Center.append(Boundary)
        else:
            Upper.append(Boundary)

plt.scatter(tot[-2],tot[-3])
plt.savefig("alpha_verNDQ.png")

for i in range(len(mean)):
    print(str(i)+"th param: {0} +{1} / -{2}".format(Center[i],Upper[i]-Center[i],-Lower[i]+Center[i]))
