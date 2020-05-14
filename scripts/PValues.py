import ElectronLifetimeTrend
from ElectronLifetimeTrend import *

import MCMC_Tools
from MCMC_Tools import *

import FormPars

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
import copy



PickleResults  = sys.argv[1]

PathContents   = PickleResults.split('/')
FileID         = PathContents[-1].split('.')[0].split('_')
PredictionFile = '/'.join(PathContents[:-2]) + '/TXTs/Prediction_' + '_'.join(FileID[1:]) + '.txt'

if len(sys.argv) > 2:
    PredictionFile = sys.argv[2]

MCMCResults = pickle.load(open(PickleResults, 'rb'))


SSELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeData_SingleScatter_ForV2Fitting.txt'
PoRnELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithPoRn.txt'
Rn222ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222.txt'



(   SSUnixtimes,
    SSUnixtimeErrors,
    SSELifeValues,
    SSELifeValueErrors) = LoadFitData('SingleScatter', PathToFile=SSELifeDataFile)

(   PoRnUnixtimes,
    PoRnUnixtimeErrors,
    PoRnELifeValues,
    PoRnELifeValueErrors) = LoadFitData('PoRn', PathToFile=PoRnELifeDataFile)

(   Rn222Unixtimes,
    Rn222UnixtimeErrors,
    Rn222ELifeValues,
    Rn222ELifeValueErrors) = LoadFitData('Rn222', PathToFile=Rn222ELifeDataFile)


print('Loading data')
(   PredictionUnixtimes,
    PredictedELifes,
    PredictedELifeLowers,
    PredictedELifeUppers,
    PredictedELifeLowerErrors,
    PredictedELifeUpperErrors) = LoadPredictions(
                                            PredictionFile,
#                                            LastPointUnixtime=LastPointUnixtime,
#                                            DaysAfterLastPoint=DaysAfterLastPoint
                                            )


PredictionInterpMedian = interp1d(PredictionUnixtimes, PredictedELifes, fill_value='extrapolate')
PredictionInterpLowerErr = interp1d(PredictionUnixtimes, np.abs(PredictedELifeLowerErrors), fill_value='extrapolate')
PredictionInterpUpperErr = interp1d(PredictionUnixtimes, PredictedELifeUpperErrors, fill_value='extrapolate')



###############################################
# calculate p-values and Bayes factors
###############################################
print('Calculating p-value and Bayes factor')
dictTrueLnLs = {}
dictDeltaLnLs = {}
dictpValues = {}
dictKs = {}
for ELifeType in ['SS', 'PoRn', 'Rn222']:
    try:
        Unixtimes        = MCMCResults[ELifeType + 'Unixtimes']
        UnixtimeErrors   = MCMCResults[ELifeType + 'UnixtimesErrors']
        ELifeValues      = MCMCResults[ELifeType + 'ELifeValues']
        ELifeValueErrors = MCMCResults[ELifeType + 'ELifeValueErrors']
    except:
        print('Unable to load ' + ELifeType + ' from pkl file, loading from data instead')
        if ELifeType == 'SS':
            (Unixtimes,
            UnixtimeErrors,
            ELifeValues,
            ELifeValueErrors) = LoadFitData('SingleScatter', PathToFile=SSELifeDataFile)
        elif ELifeType == 'PoRn':
            (Unixtimes,
            UnixtimeErrors,
            ELifeValues,
            ELifeValueErrors) = LoadFitData('PoRn', PathToFile=PoRnELifeDataFile)
        elif ELifeType == 'Rn222':
            (Unixtimes,
            UnixtimeErrors,
            ELifeValues,
            ELifeValueErrors) = LoadFitData('Rn222', PathToFile=Rn222ELifeDataFile)
    Unixtimes = np.asarray(Unixtimes)
    UnixtimeErrors = np.asarray(UnixtimeErrors)
    ELifeValues = np.asarray(ELifeValues)
    ELifeValueErrors = np.asarray(ELifeValueErrors)

    LnLs, DeltaLnLs, pValue, K = GetpValuesAndBayesFactor(
                                    Unixtimes,
                                    ELifeValues,
                                    ELifeValueErrors,
                                    PredictionInterpMedian,
                                    NumIterations=100
                                    )

    dictTrueLnLs[ELifeType] = LnLs
    dictDeltaLnLs[ELifeType] = DeltaLnLs
    dictpValues[ELifeType] = pValue
    dictKs[ELifeType] = K

#    MCMCResults[ELifeType + 'LnLs'] = LnLs
#    MCMCResults[ELifeType + 'DeltaLnLs'] = DeltaLnLs
#    MCMCResults[ELifeType + 'pValue'] = pValue
#    MCMCResults[ELifeType + 'K'] = K


###############################################
# combine into single array
###############################################
#Unixtimes = copy.deepcopy(SSUnixtimes)
#UnixtimeErrors = copy.deepcopy(SSUnixtimeErrors)
#ELifeValues = copy.deepcopy(SSELifeValues)
#ELifeValueErrors = copy.deepcopy(SSELifeValueErrors)
#TotalTrueLnLs = copy.deepcopy(dictTrueLnLs['SS'])
#TotalDeltaLnLs = copy.deepcopy(dictDeltaLnLs['SS'])


#PoRnMinUnixtime = np.min(PoRnUnixtimes)
#Rn222MinUnixtime = np.min(Rn222Unixtimes)
# Remove values that has unixtime larger than the first one in PoRnUnixTimes
# And then extend the list with Rn lists
#CutOffID = 0
#for i, unixtime in enumerate(Unixtimes):
#    if unixtime > PoRnMinUnixtime:
#        CutOffID = i
#        break
#Unixtimes = Unixtimes[0:CutOffID]
#ELifeValues = ELifeValues[0:CutOffID]
#ELifeValueErrors = ELifeValueErrors[0:CutOffID]
#TotalTrueLnLs = TotalTrueLnLs[0:CutOffID]
#TotalDeltaLnLs = TotalDeltaLnLs[0:CutOffID]

#Unixtimes.extend(PoRnUnixtimes)
#ELifeValues.extend(PoRnELifeValues)
#ELifeValueErrors.extend(PoRnELifeValueErrors)
#TotalTrueLnLs.extend(dictTrueLnLs['PoRn'])
#TotalDeltaLnLs.extend(dictDeltaLnLs['PoRn'])

#i = 0
#CutOffID = 0
#for i, unixtime in enumerate(Unixtimes):
#    if unixtime > Rn222MinUnixtime:
#        CutOffID = i
#        break

#Unixtimes = Unixtimes[0:CutOffID]
#ELifeValues = ELifeValues[0:CutOffID]
#ELifeValueErrors = ELifeValueErrors[0:CutOffID]
#TotalTrueLnLs = TotalTrueLnLs[0:CutOffID]
#TotalDeltaLnLs = TotalDeltaLnLs[0:CutOffID]
#Unixtimes.extend(Rn222Unixtimes)
#ELifeValues.extend(Rn222ELifeValues)
#ELifeValueErrors.extend(Rn222ELifeValueErrors)
#TotalTrueLnLs.extend(dictTrueLnLs['Rn222'])
#TotalDeltaLnLs.extend(dictDeltaLnLs['Rn222'])



TotalTrueLnLs = dictTrueLnLs['SS'][SSUnixtimes < PoRnUnixtimes.min()]
TotalDeltaLnLs = dictDeltaLnLs['SS'][SSUnixtimes < PoRnUnixtimes.min()]
UnixtimeErrors = SSUnixtimeErrors[SSUnixtimes < PoRnUnixtimes.min()]
ELifeValues = SSELifeValues[SSUnixtimes < PoRnUnixtimes.min()]
ELifeValueErrors = SSELifeValueErrors[SSUnixtimes < PoRnUnixtimes.min()]
Unixtimes = SSUnixtimes[SSUnixtimes < PoRnUnixtimes.min()]

TotalTrueLnLs = np.concatenate([TotalTrueLnLs, dictTrueLnLs['PoRn'][PoRnUnixtimes < Rn222Unixtimes.min()]])
TotalDeltaLnLs = np.concatenate([TotalDeltaLnLs, dictDeltaLnLs['PoRn'][PoRnUnixtimes < Rn222Unixtimes.min()]])
UnixtimeErrors = np.concatenate([UnixtimeErrors, PoRnUnixtimeErrors[PoRnUnixtimes < Rn222Unixtimes.min()]])
ELifeValues = np.concatenate([ELifeValues, PoRnELifeValues[PoRnUnixtimes < Rn222Unixtimes.min()]])
ELifeValueErrors = np.concatenate([ELifeValueErrors, PoRnELifeValueErrors[PoRnUnixtimes < Rn222Unixtimes.min()]])
Unixtimes = np.concatenate([Unixtimes, PoRnUnixtimes[PoRnUnixtimes < Rn222Unixtimes.min()]])

TotalTrueLnLs = np.concatenate([TotalTrueLnLs, dictTrueLnLs['Rn222']])
TotalDeltaLnLs = np.concatenate([TotalDeltaLnLs, dictDeltaLnLs['Rn222']])
Unixtimes = np.concatenate([Unixtimes, Rn222Unixtimes])
UnixtimeErrors = np.concatenate([UnixtimeErrors, Rn222UnixtimeErrors])
ELifeValues = np.concatenate([ELifeValues, Rn222ELifeValues])
ELifeValueErrors = np.concatenate([ELifeValueErrors, Rn222ELifeValueErrors])


print('\n=======\n')


###############################################
# print relevant info
###############################################

ScienceRunUnixtimes  = FormPars.GetScienceRunUnixtimes()

for ELifeType in dictTrueLnLs.keys():
#    print(ELifeType + ' p-value = ' +  str(dictpValues[ELifeType]) + ', K = ' +  str(dictKs[ELifeType]))
    print(ELifeType + ' p-value = %.4f, K = %.6e' %(dictpValues[ELifeType], dictKs[ELifeType]))

for ScienceRun in ScienceRunUnixtimes.keys():
    SRTrueLnL = TotalTrueLnLs[(Unixtimes > ScienceRunUnixtimes[ScienceRun][0]) &
                                (Unixtimes < ScienceRunUnixtimes[ScienceRun][1])]
    SRDeltaLnL = TotalDeltaLnLs[(Unixtimes > ScienceRunUnixtimes[ScienceRun][0]) &
                                (Unixtimes < ScienceRunUnixtimes[ScienceRun][1])]
    SRpValue = np.where(SRDeltaLnL > 0)[0].shape[0] / SRDeltaLnL.flatten().shape[0]
    SRK = np.average(np.exp(SRTrueLnL))
    print(ScienceRun + ' p-value = %.4f, K = %.6e' %(SRpValue, SRK))
#    print(ScienceRun + ' p-value = ' + str(SRpValue) + ', K = ' + str(SRK))

pValueTotal = np.where(TotalDeltaLnLs > 0)[0].shape[0] / TotalDeltaLnLs.flatten().shape[0]
KTotal = np.average(np.exp(TotalTrueLnLs))
#print('Total p-value = ' + str(pValueTotal) + ', K = ' + str(KTotal))
print('Total p-value = %.4f, K = %.6e' %(pValueTotal, KTotal))
