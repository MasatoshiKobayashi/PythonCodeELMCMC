import ElectronLifetimeTrend
from ElectronLifetimeTrend import *

import MCMC_Tools
from MCMC_Tools import *

import FormPars

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv

import Fit_func2
from  Fit_func2 import *

import matplotlib.pyplot as plt

def GetNewPlot(xtitle,ytitle):
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax = fig.add_subplot(1,1,1)
    plt.tight_layout()
    plt.ylabel(ytitle, fontsize=18)
    plt.xlabel(xtitle, fontsize=18)
    return ax


if len(sys.argv)<2:
    print("======= Syntax ========")
    print("python PredictElectronLifetime.py")
    print("<Historian file>")
    print("<Fit result pickle>")
    print("<Prediction txt output>")
    print("<burn-in iteraction cut>")
    print("NOTICE: do remember to add \"-r\" at the end")
    exit()

SSELifeDataFile ='/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeData_SingleScatter_ForV2Fitting.txt'
PoRnELifeDataFile =  '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithPoRn.txt'
Po218ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithPo218_1day_new_wobad.txt'
Rn220ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn220_1day_new_wobad.txt'
Rn222ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222_1day_new_wobad.txt'
FileParams = '/dali/lgrandi/kobayashi1/xenon/analysis/ElectronLifetime/ElectronLifetime/PythonCodeELMCMCndna_semi_final24_2/TEMP_Param.txt'


MinUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMinTimeStamp())
MaxUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMaxTimeStamp()) 
#parser = argparse.ArgumentParser()
#parser.add_argument('--min_unixtime', type=int, help='Minimum unixtime', default=MinUnixTime)
#parser.add_argument('--max_unixtime', type=int, help='Maximum unixtime', default=MaxUnixTime)
#parser.add_argument('--include_firstss',type=int, choices=[0, 1], help='Include first ss combined fits', default=1)
#
#args = parser.parse_args()
#
#IncludeFirstSS               = args.include_firstss
#MinUnixTime                  = args.min_unixtime
#MaxUnixTime                  = args.max_unixtime
#verbose                      = args.verbose



HistorianFile = sys.argv[1]
FitResultInput = sys.argv[2]
PredictionOutputFile = sys.argv[3]
BurnInCutOff = int(sys.argv[4])


NumOfInterpolation = 1000
NumOfTrials = 400


# setting the parameters
MinUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMinTimeStamp())
#MaxUnixTime = GetUnixTimeFromTimeStamp(FormPars.GetMaxTimeStamp()) 
MaxUnixTime=1485993600
MaxUnixTime=1517529600
MaxUnixTime=1533081600 
IncludeFirstSS=0
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
samples_cut = GetBurnInCutoffSamplesV2(sampler.__dict__['_chain'], int(BurnInCutOff), int(BurnInCutOff)+199)

#####################################
## Calculate the best fit from MCMC results
#####################################
mean = np.average(samples_cut, axis=0)
print(mean)


Pars, IfSth,parindex=FormPars.FormPars(mean)
#MaxUnixTime = LastPointUnixTime + 60.*3600.*24. # 2 month after the last data point
# The main Light yield Trend
#pElectronLifetimeTrend = MyElectronLifetimeTrend(HistorianFile, MinUnixTime, MaxUnixTime, Pars)

SetGlobalPars(nwalkers,niterations,28,1,MinUnixTime,MaxUnixTime,0,IncludeFirstSS) 
x0, x0_steps =SetFiles(HistorianFile,FitOutput,PredictionOutputFile,FileParams)
print("Setting Files...")
ElectronLifetimeData=SetELifeData(SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
ElectronLifetimeData=SetELifeDataDiv(SSELifeDataFile,PoRnELifeDataFile,Po218ELifeDataFile,Rn220ELifeDataFile,Rn222ELifeDataFile)
print("Done!")


DUnixTimes=ElectronLifetimeData['UnixTimes'] 
DValues=ElectronLifetimeData['Values'] 
DValueErrors=ElectronLifetimeData['ValueErrors'] 


# get the graphs 
UnixTimes = np.linspace(MinUnixTime, MaxUnixTime, NumOfInterpolation)
# include times of discontinuity
ImpactfulUnixtimes = FormPars.GetImpactfulUnixtimes()
for ImpactfulUnixtime in ImpactfulUnixtimes:
	if ImpactfulUnixtime-1 not in UnixTimes:
		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime-1]), ImpactfulUnixtime-1)
	if ImpactfulUnixtime not in UnixTimes:
		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime]), ImpactfulUnixtime)
	if ImpactfulUnixtime+1 not in UnixTimes:
		UnixTimes = np.insert(UnixTimes, len(UnixTimes[UnixTimes < ImpactfulUnixtime+1]), ImpactfulUnixtime+1)

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

## Initial the 1 sigma lower/upper of the trend
#Trends = []
#for i in range(len(UnixTimes)):
#    Trends.append([])
#
#
## pickup randomly the pars in the MCMC walker
#Pars_Trials = PickupMCMCPosteriors(samples_cut, NumOfTrials)
#for i, pars_random in enumerate(Pars_Trials):
#    print("i = "+str(i))
#    pars, IfSth = FormPars.FormPars(pars_random)
#    pElectronLifetimeTrend.SetParameters(pars)
#    for j, unixtime in enumerate(UnixTimes):
#        Trends[j].append(pElectronLifetimeTrend.GetElectronLifetime(unixtime))


Pars_Trials = PickupMCMCPosteriors(samples_cut, NumOfTrials)
#print(UnixTimes)
Trends,TrendsImpurityGXe,TrendsImpurityLXe,Nlost,UnixTimes_chi2,Trends_chi2,Trends_chi2_tot,Trends_Rchi2,Trends_Rchi2_tot=Randomize(UnixTimes,Pars_Trials)
#alphaEF0s = []
#for i, pars_random in enumerate(Pars_Trials):
#    if(i%50==0):
#        print("i = "+str(i))
#    pars, IfSth, parindex = FormPars.FormPars(pars_random)
#    alphaSS,alphaPAX,alphaPoRn,alphaEFB, alphaEF0,alphaEF1,alphaAttach = pars[-1][0],pars[-1][1],pars[-1][2],pars[-1][3],pars[-1][4],pars[-1][5],pars[-1][6]
#    alphaEF0s.append(alphaEF0)


print('{} Trials lost integratoins'.format(Nlost))
StandardDeviation1Sigma = [15.4, 50., 84.6]
Taus = []
LowerBoundaries = []
UpperBoundaries = []
ITaus = []
ILowerBoundaries = []
IUpperBoundaries = []
IgTaus = []
IgLowerBoundaries = []
IgUpperBoundaries = []
chi2Taus = []
chi2LowerBoundaries = []
chi2UpperBoundaries = []
Rchi2Taus = []
Rchi2LowerBoundaries = []
Rchi2UpperBoundaries = []

for i, deviation in enumerate(StandardDeviation1Sigma):
    for Values in Trends:
        #print(Values)
        Boundarya = PercentileWithInf(Values, deviation)
        if(np.isscalar(Boundarya)):
            Boundary=Boundarya
        else:
            Boundary=Boundarya[0]
        if i==0:
            LowerBoundaries.append(Boundary)
        elif i==1:
            Taus.append(Boundary)
        else:
            UpperBoundaries.append(Boundary)

    for Values in TrendsImpurityLXe:
        #print(Values)
        Boundarya = PercentileWithInf(Values, deviation)
        if(np.isscalar(Boundarya)):
            IBoundary=Boundarya
        else:
            IBoundary=Boundarya[0]
        if i==0:
            ILowerBoundaries.append(IBoundary)
        elif i==1:
            ITaus.append(IBoundary)
        else:
            IUpperBoundaries.append(IBoundary)

    for Values in TrendsImpurityGXe:
        #print(Values)
        Boundarya = PercentileWithInf(Values, deviation)
        if(np.isscalar(Boundarya)):
            IgBoundary=Boundarya
        else:
            IgBoundary=Boundarya[0]
        if i==0:
            IgLowerBoundaries.append(IgBoundary)
        elif i==1:
            IgTaus.append(IgBoundary)
        else:
            IgUpperBoundaries.append(IgBoundary)

    for Values in Trends_chi2:
        #print(Values)
        Boundarya = PercentileWithInf(Values, deviation)
        if(np.isscalar(Boundarya)):
            chi2Boundary=Boundarya
        else:
            chi2Boundary=Boundarya[0]
        if i==0:
            chi2LowerBoundaries.append(chi2Boundary)
        elif i==1:
            chi2Taus.append(chi2Boundary)
        else:
            chi2UpperBoundaries.append(chi2Boundary)

    for Values in Trends_Rchi2:
        #print(Values)
        Boundarya = PercentileWithInf(Values, deviation)
        if(np.isscalar(Boundarya)):
            Rchi2Boundary=Boundarya
        else:
            Rchi2Boundary=Boundarya[0]
        if i==0:
            Rchi2LowerBoundaries.append(Rchi2Boundary)
        elif i==1:
            Rchi2Taus.append(Rchi2Boundary)
        else:
            Rchi2UpperBoundaries.append(Rchi2Boundary)



#print(MaximumELifes)


######### moved from bottom because terminal cannot show root plots ##########
fout = open(PredictionOutputFile+".txt", 'w')
for unixtime, tau, lower, upper in zip(UnixTimes, Taus, LowerBoundaries, UpperBoundaries):
    fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
    fout.write("\n")
fout.close()
#######################################################################

fout = open(PredictionOutputFile+"_data.txt", 'w')
for unixtime, lt,lt2 in zip(DUnixTimes,DValues,DValueErrors):
    fout.write("{0}\t\t{1}\t\t{2}".format(unixtime,lt,lt2))
    fout.write("\n")
fout.close()

######### moved from bottom because terminal cannot show root plots ##########
fout = open(PredictionOutputFile+"_Ig.txt", 'w')
for unixtime, tau, lower, upper in zip(UnixTimes, IgTaus, IgLowerBoundaries, IgUpperBoundaries):
    fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
    fout.write("\n")
fout.close()
######### moved from bottom because terminal cannot show root plots ##########
fout = open(PredictionOutputFile+"_Il.txt", 'w')
for unixtime, tau, lower, upper in zip(UnixTimes, ITaus, ILowerBoundaries, IUpperBoundaries):
    fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
    fout.write("\n")
fout.close()
######### moved from bottom because terminal cannot show root plots ##########
fout = open(PredictionOutputFile+"_chi2.txt", 'w')
for unixtime, tau, lower, upper in zip(UnixTimes_chi2, chi2Taus, chi2LowerBoundaries, chi2UpperBoundaries):
    fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
    fout.write("\n")
fout.close()

ax1=GetNewPlot("chi2","entries")
plt.hist(Trends_chi2_tot,bins=1000,range=(500,1500))
plt.savefig(PredictionOutputFile.replace("TXTs","Figures")+"_chi2.png")

fout = open(PredictionOutputFile+"_Rchi2.txt", 'w')
for unixtime, tau, lower, upper in zip(UnixTimes_chi2, Rchi2Taus, Rchi2LowerBoundaries, Rchi2UpperBoundaries):
    fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
    fout.write("\n")
fout.close()

ax1=GetNewPlot("Rchi2","entries")
plt.hist(Trends_Rchi2_tot,bins=50,range=(0,5))
plt.savefig(PredictionOutputFile.replace("TXTs","Figures")+"_Reducedchi2.png")


#ax1=GetNewPlot("chi2","alpha")
#plt.scatter(Trends_chi2_tot,alphaEF0s)
#plt.savefig(PredictionOutputFile.replace("TXTs","Figures")+"_alpha_vs_chi2.png")
