import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import gridspec
import datetime as dt
import time
import pickle
import sys
import glob
import os
from collections import OrderedDict
from MCMC_Tools import *
import FormPars

if len(sys.argv)<2:
    print("======== Syntax: =======")
    print("python DrawPlotForElectronLifetimePrediction.py .....")
    print("< elife data txt file> ")
    print("<PoRn elife data txt file>")
    print("<Kr83 elife data txt file>")
    print("< prediction txt file> ")
    print("< days to show after last data point >")
    print("<save fig name (rel.)>")
    exit()

ELifeDataFile = sys.argv[1]
PoRnELifeDataFile = sys.argv[2]
Kr83ELifeDataFile = sys.argv[3]
PredictionFile = sys.argv[4]
DaysAfterLastPoint = float(sys.argv[5])
FigureSaveName = sys.argv[6]




#XE1T.CRY_110_PT112_PINS_AI.PI
#XE1T.CRY_PI110_PT111_PINS_AI.PI
#XE1T.CRY_CHILL102_TT162_TMON_AI.PI

VariablesToPlot = OrderedDict()
VariablesToPlot['XE1T.CRY_PT101_PCHAMBER_AI.PI']       = 'Detector pressure (cryostat)'
VariablesToPlot['XE1T.CRY_PT103_PCHAMBER_AI.PI']       = 'detector pressure'
VariablesToPlot['XE1T.CRY_PI110_PT113_PINS_AI.PI']     = 'Insulatoin vacuum (cryogenics)'
VariablesToPlot['XE1T.CRY_TIC111_TE113_TVESSEL_AI.PI'] = 'Tower inner vessel temperature'
VariablesToPlot['XE1T.CRY_TE104_TCRYOBELL_AI.PI']      = 'Cryostat below bell temperature (LXe)'
VariablesToPlot['XE1T.CRY_TE106_TINSBELL_AI.PI']       = 'inside bell temperature (GXe)'
VariablesToPlot['XE1T.CRY_TE107_TCRYOTOP_AI.PI']       = 'cryostat top temperature (GXe)'
VariablesToPlot['TPC_Monitor_Voltage']       	       = 'TPC voltage'

for key in VariablesToPlot:
    print('|  ' + key + '  |  ' + VariablesToPlot[key] + '  |')

yLimsToPlot = OrderedDict()
yLimsToPlot['XE1T.CRY_PT101_PCHAMBER_AI.PI']       = [1.925, 1.945]
yLimsToPlot['XE1T.CRY_PT103_PCHAMBER_AI.PI']       = [1.925, 1.945]
yLimsToPlot['XE1T.CRY_PI110_PT113_PINS_AI.PI']     = [2.1e-6, 4.4e-6]
yLimsToPlot['XE1T.CRY_TIC111_TE113_TVESSEL_AI.PI'] = [15.1, 17.4]
yLimsToPlot['XE1T.CRY_TE104_TCRYOBELL_AI.PI']      = [-96.01, -95.63]
yLimsToPlot['XE1T.CRY_TE106_TINSBELL_AI.PI']       = [-96.13, -95.62]
yLimsToPlot['XE1T.CRY_TE107_TCRYOTOP_AI.PI']       = [-89.32, -88.31]
yLimsToPlot['TPC_Monitor_Voltage']       	   = [-0.1, 20.]


#VariableList = 'InvestigateLifetimeDrop'
VariableList = 'ImpurityUpdated'
#d = pickle.load(open('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/SlowControl/PickleFiles/%s_161101_to_171023.p' %VariableList, 'rb'))
d = pickle.load(open('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/SlowControl/PickleFiles/%s_160501_to_171122.pkl' %VariableList, 'rb'))


ScienceRunUnixtimes  = FormPars.GetScienceRunUnixtimes()

Xe40kevELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithXe40keV.txt'
Xe129mELifeDataFile  = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithXe129m.txt'
Xe131mELifeDataFile  = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithXe131m.txt'
Rn222ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222.txt'
Po218ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithPo218.txt'
Rn220ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn220.txt'

ShowResiduals     = False
ShowResiduals     = True
FillScienceRuns   = False
DisplayScienceRun = False

PlotKrEvolution   = True
PlotKrEvolution   = False


PathContents         = PredictionFile.split('/')
PathToKrFolder       = '/'.join(PathContents[:-1]) + '/Kr83m/'
Kr83PredictionFile   = PathToKrFolder + 'Kr_' + PathContents[-1]

if not os.path.exists(Kr83PredictionFile) and PlotKrEvolution:
    FileList           = PathToKrFolder + '*'
    SortedFiles        = sorted(glob.iglob(FileList), key=os.path.getctime, reverse=True)
    Kr83PredictionFile = SortedFiles[0]



#######################################
### Get elife data
#######################################
(   UnixTimes,
    UnixTimeErrors,
    ELifeValues,
    ELifeValueErrors) = LoadFitData('SingleScatter', PathToFile=ELifeDataFile)

(   PoRnUnixtimes,
    PoRnUnixtimeErrors,
    PoRnELifeValues,
    PoRnELifeValueErrors) = LoadFitData('PoRn', PathToFile=PoRnELifeDataFile)

(   Rn222Unixtimes,
    Rn222UnixtimeErrors,
    Rn222ELifeValues,
    Rn222ELifeValueErrors) = LoadFitData('Rn222', PathToFile=Rn222ELifeDataFile)

(   Po218Unixtimes,
    Po218UnixtimeErrors,
    Po218ELifeValues,
    Po218ELifeValueErrors) = LoadFitData('Po218', PathToFile=Po218ELifeDataFile)

(   Rn220Unixtimes,
    Rn220UnixtimeErrors,
    Rn220ELifeValues,
    Rn220ELifeValueErrors) = LoadFitData('Rn220', PathToFile=Rn220ELifeDataFile)


(   Xe40kevUnixtimes,
    Xe40kevUnixtimeErrors,
    Xe40kevELifeValues,
    Xe40kevELifeValueErrors) = LoadFitData('Xe129', PathToFile=Xe40kevELifeDataFile)

(   Xe129mUnixtimes,
    Xe129mUnixtimeErrors,
    Xe129mELifeValues,
    Xe129mELifeValueErrors) = LoadFitData('Xe129m', PathToFile=Xe129mELifeDataFile)


(   Xe131mUnixtimes,
    Xe131mUnixtimeErrors,
    Xe131mELifeValues,
    Xe131mELifeValueErrors) = LoadFitData('Xe131m', PathToFile=Xe131mELifeDataFile)

(   KrUnixtimes,
    KrUnixtimeErrors,
    KrELifeValues,
    KrELifeValueErrors) = LoadFitData('Kr83', PathToFile=Kr83ELifeDataFile)


FirstPointUnixTime = UnixTimes[0]
LastPointUnixtime = UnixTimes[len(UnixTimes)-1]
LastPointUnixtime = PoRnUnixtimes[-1]
LastPointUnixtime = Rn222Unixtimes[-1]




CutID = 0
for i, unixtime in enumerate(UnixTimes):
    if unixtime>PoRnUnixtimes[0]:
        CutID = i
        break

UnixTimes = UnixTimes[:CutID]
UnixTimeErrors = UnixTimeErrors[:CutID]
ELifeValues = ELifeValues[:CutID]
ELifeValueErrors = ELifeValueErrors[:CutID]

CutID = 0
for i, unixtime in enumerate(PoRnUnixtimes):
    if unixtime>Rn222Unixtimes[0]:
        CutID = i
        break

PoRnUnixtimes = PoRnUnixtimes[:CutID]
PoRnUnixtimeErrors = PoRnUnixtimeErrors[:CutID]
PoRnELifeValues = PoRnELifeValues[:CutID]
PoRnELifeValueErrors = PoRnELifeValueErrors[:CutID]



#######################################
## Get the prediction lists
#######################################

(   PredictionUnixtimes,
    Rn222PredictedELifes,
    Rn222PredictedELifeLowers,
    Rn222PredictedELifeUppers,
    Rn222PredictedELifeLowerErrors,
    Rn222PredictedELifeUpperErrors) = LoadPredictions(
                                                        PredictionFile,
                                                        LastPointUnixtime=LastPointUnixtime,
                                                        DaysAfterLastPoint=DaysAfterLastPoint
                                                        )

(   Rn222LowerFitUncertainty,
    Rn222UpperFitUncertainty) = GetPredictionUncertainties(
                                                            PredictionUnixtimes,
                                                            Rn222PredictedELifes,
                                                            Rn222PredictedELifeLowerErrors,
                                                            Rn222PredictedELifeUpperErrors
                                                            )

Rn222PredictionInterpolator = interp1d(PredictionUnixtimes, Rn222PredictedELifes)

if PlotKrEvolution:
    print('\nLoading ' + Kr83PredictionFile + ' for Kr83m lifetime evolution\n')
    (   KrPredictionUnixtimes,
        Kr83PredictedELifes,
        Kr83PredictedELifeLowers,
        Kr83PredictedELifeUppers,
        Kr83PredictedELifeLowerErrors,
        Kr83PredictedELifeUpperErrors) = LoadPredictions(
                                                            Kr83PredictionFile,
                                                            LastPointUnixtime=LastPointUnixtime,
                                                            DaysAfterLastPoint=DaysAfterLastPoint
                                                            )



    #Kr83PredictionInterpolator = interp1d(PredictionUnixtimes, Kr83PredictedELifes)
    Kr83PredictionInterpolator = interp1d(KrPredictionUnixtimes, Kr83PredictedELifes)

    (   Kr83LowerFitUncertainty,
        Kr83UpperFitUncertainty) = GetPredictionUncertainties(
                                                                KrPredictionUnixtimes,
                                                                Kr83PredictedELifes,
                                                                Kr83PredictedELifeLowerErrors,
                                                                Kr83PredictedELifeUpperErrors
                                                                )





###################################
## Get the residual of the data points
###################################
(   ELifeValueDeviations,
    ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            UnixTimes,
                                                            ELifeValues,
                                                            ELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

(   PoRnELifeValueDeviations,
    PoRnELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            PoRnUnixtimes,
                                                            PoRnELifeValues,
                                                            PoRnELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

(   Rn220ELifeValueDeviations,
    Rn220ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Rn220Unixtimes,
                                                            Rn220ELifeValues,
                                                            Rn220ELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

(   Rn222ELifeValueDeviations,
    Rn222ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Rn222Unixtimes,
                                                            Rn222ELifeValues,
                                                            Rn222ELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

(   Po218ELifeValueDeviations,
    Po218ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Po218Unixtimes,
                                                            Po218ELifeValues,
                                                            Po218ELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

(   Rn220ELifeValueDeviations,
    Rn220ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Rn220Unixtimes,
                                                            Rn220ELifeValues,
                                                            Rn220ELifeValueErrors,
                                                            Rn222PredictionInterpolator
                                                            )

TotalUnixtimes        = UnixTimes            + PoRnUnixtimes            + Rn222Unixtimes
TotalELifeValues      = ELifeValues          + PoRnELifeValues          + Rn222ELifeValues
TotalELifeValueErrors = ELifeValueErrors     + PoRnELifeValueErrors     + Rn222ELifeValueErrors
TotalELifeDeviations  = ELifeValueDeviations + PoRnELifeValueDeviations + Rn222ELifeValueDeviations

################
### Get Biases
################
MeanBias = {}
RMSBias = {}
MeanBias['Total'], RMSBias['Total'] = GetBiases(TotalUnixtimes, TotalELifeDeviations)
print('Total bias: %.2f,\t Total RMS: %.2f' %(MeanBias['Total'], RMSBias['Total']))

for ScienceRun in ScienceRunUnixtimes.keys():
    MeanBias[ScienceRun], RMSBias[ScienceRun] = GetBiases(TotalUnixtimes, TotalELifeDeviations,
                                                          StartUnixtime=ScienceRunUnixtimes[ScienceRun][0],
                                                          EndUnixtime=ScienceRunUnixtimes[ScienceRun][1])

    print(ScienceRun + 'bias:   %.2f,\t '%(MeanBias[ScienceRun]) + ScienceRun + ' RMS:   %.2f' %(RMSBias[ScienceRun]))


###################################
## Calculate the uncertainties in the first science run
###################################


###################################
## convert unixtimes to dates
###################################
Dates = [dt.datetime.fromtimestamp(ts) for ts in UnixTimes]
PoRnDates = [dt.datetime.fromtimestamp(ts) for ts in PoRnUnixtimes]
Po218Dates = [dt.datetime.fromtimestamp(ts) for ts in Po218Unixtimes]
Rn220Dates = [dt.datetime.fromtimestamp(ts) for ts in Rn220Unixtimes]
Rn222Dates = [dt.datetime.fromtimestamp(ts) for ts in Rn222Unixtimes]
KrDates = [dt.datetime.fromtimestamp(ts) for ts in KrUnixtimes]
Xe40kevDates = [dt.datetime.fromtimestamp(ts) for ts in Xe40kevUnixtimes]
Xe129mDates = [dt.datetime.fromtimestamp(ts) for ts in Xe129mUnixtimes]
Xe131mDates = [dt.datetime.fromtimestamp(ts) for ts in Xe131mUnixtimes]

DateErrorLowers = []
DateErrorUppers = []
for ts, ts_err in zip(UnixTimes, UnixTimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    DateErrorLowers.append( date_err_lower )
    DateErrorUppers.append( date_err_upper )

PoRnDateErrorLowers = []
PoRnDateErrorUppers = []
for ts, ts_err in zip(PoRnUnixtimes, PoRnUnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    PoRnDateErrorLowers.append( date_err_lower )
    PoRnDateErrorUppers.append( date_err_upper )

Po218DateErrorLowers = []
Po218DateErrorUppers = []
for ts, ts_err in zip(Po218Unixtimes, Po218UnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Po218DateErrorLowers.append( date_err_lower )
    Po218DateErrorUppers.append( date_err_upper )

Rn220DateErrorLowers = []
Rn220DateErrorUppers = []
for ts, ts_err in zip(Rn220Unixtimes, Rn220UnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Rn220DateErrorLowers.append( date_err_lower )
    Rn220DateErrorUppers.append( date_err_upper )

Rn222DateErrorLowers = []
Rn222DateErrorUppers = []
for ts, ts_err in zip(Rn222Unixtimes, Rn222UnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Rn222DateErrorLowers.append( date_err_lower )
    Rn222DateErrorUppers.append( date_err_upper )

Xe40kevDateErrorLowers = []
Xe40kevDateErrorUppers = []
for ts, ts_err in zip(Xe40kevUnixtimes, Xe40kevUnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Xe40kevDateErrorLowers.append( date_err_lower )
    Xe40kevDateErrorUppers.append( date_err_upper )

Xe129mDateErrorLowers = []
Xe129mDateErrorUppers = []
for ts, ts_err in zip(Xe129mUnixtimes, Xe129mUnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Xe129mDateErrorLowers.append( date_err_lower )
    Xe129mDateErrorUppers.append( date_err_upper )

Xe131mDateErrorLowers = []
Xe131mDateErrorUppers = []
for ts, ts_err in zip(Xe131mUnixtimes, Xe131mUnixtimeErrors):
    date = dt.datetime.fromtimestamp(ts)
    date_err_lower = date - dt.datetime.fromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.fromtimestamp(ts + ts_err) - date
    Xe131mDateErrorLowers.append( date_err_lower )
    Xe131mDateErrorUppers.append( date_err_upper )
#UnixTimes2 = np.asarray(UnixTimes2)
#PredictedELifes = np.asarray(PredictedELifes)
#PredictedELifeLowers = np.asarray(PredictedELifeLowers)
#PredictedELifeUppers = np.asarray(PredictedELifeUppers)
#Dates2 = [dt.datetime.fromtimestamp(ts) for ts in UnixTimes2[UnixTimes2 < 1484731512]]
Dates2 = [dt.datetime.fromtimestamp(ts) for ts in PredictionUnixtimes]
if PlotKrEvolution:
    Dates3 = [dt.datetime.fromtimestamp(ts) for ts in KrPredictionUnixtimes]


##############################
## Draw plot
##############################


fig = plt.figure(figsize=(35.0, 70.0))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

gs1 = gridspec.GridSpec(10,1)
if ShowResiduals:
    ax = plt.subplot(gs1[0:2,:])
    ax2 = plt.subplot(gs1[2:3,:], sharex=ax)
    axs = [plt.subplot(gs1[3:4,:], sharex=ax)]
    axs.append(plt.subplot(gs1[4:5,:], sharex=ax))
    axs.append(plt.subplot(gs1[5:6,:], sharex=ax))
    axs.append(plt.subplot(gs1[6:7,:], sharex=ax))
    axs.append(plt.subplot(gs1[7:8,:], sharex=ax))
    axs.append(plt.subplot(gs1[8:9,:], sharex=ax))
    axs.append(plt.subplot(gs1[9:,:], sharex=ax))
else:
    ax = plt.subplot(gs1[0:3,:])

#xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)


CathodeVoltages = FormPars.GetCathodeVoltages()

# plot times when voltage is not 0 kV, otherwise fill
for CathodeVoltage in CathodeVoltages:
    if CathodeVoltage[1][0] == 0 and CathodeVoltage[1][1] == 0:
#        ax.fill_between(Dates2, 0, 650, color='y', alpha=0.5, label=r'$V_{C} = 0$ kV')
        continue

    Dates2 = [dt.datetime.fromtimestamp(ts) for ts in PredictionUnixtimes
                if(ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    Rn222ELifesToPlot       = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifes)
                                if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]
    Rn222ELifesLowToPlot    = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifeLowers)
                                if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]
    Rn222ELifesUpToPlot     = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifeUppers)
                                if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    if PlotKrEvolution:
        KrDates2 = [dt.datetime.fromtimestamp(ts) for ts in KrPredictionUnixtimes
                    if(ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

        Kr83ELifesToPlot       = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifes)
                                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]
        Kr83ELifesLowToPlot    = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifeLowers)
                                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]
        Kr83ELifesUpToPlot     = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifeUppers)
                                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

#LowerFitUncertainty
#UpperFitUncertainty

    Rn222ELifesLowErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222LowerFitUncertainty)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    Rn222ELifesUpErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222UpperFitUncertainty)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]


    ax.plot(
                Dates2,
                Rn222ELifesToPlot,
                linewidth=2.,
                color = 'r',
#                color = 'gold',
                label='Best-fit trend',
               )
    ax.fill_between(
                             Dates2,
                             Rn222ELifesLowToPlot,
                             Rn222ELifesUpToPlot,
                             color='b',
#                             color='c',
                             label=r'$\pm 1 \sigma$ C.L. region',
                             alpha=0.5,
                            )

    if PlotKrEvolution:
        ax.plot(
                    KrDates2,
                    Kr83ELifesToPlot,
                    linewidth=2.,
                    color = 'y',
#                   color = 'gold',
                    label='Best-fit trend',
                   )
        ax.fill_between(
                                 KrDates2,
                                 Kr83ELifesLowToPlot,
                                 Kr83ELifesUpToPlot,
                                 color='g',
#                                color='c',
                                 label=r'$\pm 1 \sigma$ C.L. region',
                                 alpha=0.5,
                                )
    if ShowResiduals:
        ax2.hlines(0, min(Dates2), max(Dates2), color='deeppink', linewidth=2)
        ax2.fill_between(
                                 Dates2,
                                 Rn222ELifesLowErrToPlot,
                                 Rn222ELifesUpErrToPlot,
                                 color='b',
                                 alpha=0.5,
                                )


#ax.errorbar(Dates, ELifeValues, xerr=[DateErrorLowers,DateErrorUppers],
#            yerr=[ELifeValueErrors,ELifeValueErrors], fmt='o', color='k',
#            label="S2/S1 method")

ax.errorbar(Xe40kevDates, Xe40kevELifeValues,  xerr = [Xe40kevDateErrorLowers,Xe40kevDateErrorUppers],
            yerr=[Xe40kevELifeValueErrors,Xe40kevELifeValueErrors], fmt='o', color='deepskyblue',
            marker='*', markersize=10, label='$\\rm{^{129}Xe}\,\,\, 39.6\,\, keV$')

ax.errorbar(KrDates, KrELifeValues, yerr = [KrELifeValueErrors, KrELifeValueErrors],
            fmt = 'o', color = 'g', label ='$\\rm{^{83m}Kr}\,\,\, 41.6\,\, keV$')

#ax.errorbar(Xe129mDates, Xe129mELifeValues,  xerr = [Xe129mDateErrorLowers,Xe129mDateErrorUppers],
#            yerr=[Xe129mELifeValueErrors,Xe129mELifeValueErrors], fmt='o', color='darkmagenta',
#            marker='*', markersize=10, label='$\\rm{^{131m}Xe}\,\,\, 163.9\,\, keV$')
#
#ax.errorbar(Xe131mDates, Xe131mELifeValues,  xerr = [Xe131mDateErrorLowers,Xe131mDateErrorUppers],
#            yerr=[Xe131mELifeValueErrors,Xe131mELifeValueErrors], fmt='o', color='darkorange',
#            marker='*', markersize=10, label='$\\rm{^{129m}Xe}\,\,\, 236.1\,\, keV$')

ax.errorbar(PoRnDates, PoRnELifeValues,  xerr = [PoRnDateErrorLowers,PoRnDateErrorUppers],
            yerr=[PoRnELifeValueErrors,PoRnELifeValueErrors], fmt='o', color='deeppink',
            marker='v', label='$\\rm{^{218}Po}/^{222}Rn}$')

ax.errorbar(Po218Dates, Po218ELifeValues,  xerr = [Po218DateErrorLowers,Po218DateErrorUppers],
            yerr=[Po218ELifeValueErrors,Po218ELifeValueErrors], fmt='o', color='c',
            marker='v', markersize=8, label='$\\rm{^{218}Po}\,\,\, 6.114\,\, MeV$')

ax.errorbar(Rn222Dates, Rn222ELifeValues,  xerr = [Rn222DateErrorLowers,Rn222DateErrorUppers],
            yerr=[Rn222ELifeValueErrors,Rn222ELifeValueErrors], fmt='o', color='gold',
            marker='v', markersize=8, label='$\\rm{^{222}Rn}\,\,\, 5.590\,\, MeV$')

ax.errorbar(Rn220Dates, Rn220ELifeValues,  xerr = [Rn220DateErrorLowers,Rn220DateErrorUppers],
            yerr=[Rn220ELifeValueErrors,Rn220ELifeValueErrors], fmt='o', color='orangered',
            marker='v', markersize=8, label='$\\rm{^{220}Rn}\,\,\, 6.404\,\, MeV$')


#TotalDates = [dt.datetime.fromtimestamp(ts) for ts in TotalUnixtimes]
#ax.errorbar(TotalDates, TotalELifeValues,
#            yerr=[TotalELifeValueErrors,TotalELifeValueErrors], fmt='o', color='r',
#            label="all")



# plot the vertical lines for system change
ax.axvline( x=dt.datetime.fromtimestamp(1465937520), # first power outage
                    ymin = 0,
                    ymax = 700, 
                    linestyle = "--",
                    linewidth=3,
                    color='k',
                   )
ax.axvline( x=dt.datetime.fromtimestamp(1468597800), # LN2 test. PTR1 warm-up
                    ymin = 0,
                    ymax = 700, 
                    linestyle = "--",
                    linewidth=3,
                    color='k',
                   )
ax.axvline( x=dt.datetime.fromtimestamp(1484731512), # earthquake
                    ymin = 0,
                    ymax = 700, 
                    linestyle = "--",
                    linewidth=3,
                    color='k',
                   )



# fill the region
Xs = [
          dt.datetime.fromtimestamp(1471880000),
          dt.datetime.fromtimestamp(1472800000)
         ]
YLs = [0, 0]
YUs = [700, 700]
ax.fill_between(Xs, YLs, YUs, color='coral', alpha=0.7)
Xs = [
          dt.datetime.fromtimestamp(1475180000),
          dt.datetime.fromtimestamp(1475680000)
         ]
ax.fill_between(Xs, YLs, YUs, color='m', alpha=0.3)

if FillScienceRuns:
    for ScienceRun in ScienceRunUnixtimes:
        Xs = [
                  dt.datetime.fromtimestamp(ScienceRunUnixtimes[ScienceRun][0]),
                  dt.datetime.fromtimestamp(ScienceRunUnixtimes[ScienceRun][1])
                 ]
        YLs = [0, 0]
        YUs = [700, 700]
        ax.fill_between(Xs, YLs, YUs, color='coral', alpha=0.5)



# plot the text
ax.text( # Power outage
            dt.datetime.fromtimestamp(1465937520+2.*3600.*24.), 
            200., 
            'PTR warm-up',
            color='k',
            size=22.,
            rotation='vertical',
            )
ax.text( # LN2 test, PTR warm up
            dt.datetime.fromtimestamp(1468597800+2.*3600.*24.), 
            450., 
            'LN2 cooling test; PTR warm-up',
            color='k',
            size=22.,
            rotation='vertical',
            )
#ax.text( # Earthquake @ 01/18/2017
#            dt.datetime.fromtimestamp(1484731512+2.*3600.*24.), 
#            450., 
#            'Earthquake @ 01/18/17',
#            color='k',
#            size=22.,
#            rotation='vertical',
#            )
ax.text( # Gas-only circulation
            dt.datetime.fromtimestamp(1471880000-7.*3600.*24.), 
            680., 
            'Gas-only circulation',
            color='coral',
            size=22.,
            #rotation='vertical',
            )
ax.text(dt.datetime.fromtimestamp(1471880000), 580+20, "20 SLPM", color='coral', size=22.)
ax.text( # PUR upgrade
            dt.datetime.fromtimestamp(1475180000-5.*3600.*24.), 
            700., 
            'PUR upgrade',
            color='m',
            size=22.,
            alpha=0.5,
            #rotation='vertical',
            )


# text the flow rate
ax.text( dt.datetime.fromtimestamp(1464000000), 580+40., "$\sim$ 40 SLPM", size=20.,color='k')
ax.text( dt.datetime.fromtimestamp(1466500000), 580+20, "$\sim$ 55 SLPM", size=20.,color='k')
ax.text( dt.datetime.fromtimestamp(1469500000), 580+40, "45 - 50 SLPM", size=20.,color='k')
ax.text( dt.datetime.fromtimestamp(1473500000), 580+40, "$\sim$ 40 SLPM", size=20.,color='k')
ax.text( dt.datetime.fromtimestamp(1475700000), 580+20, "$\sim$ 54 SLPM", size=20.,color='k')


Dates2 = [dt.datetime.fromtimestamp(ts) for ts in PredictionUnixtimes]
if PlotKrEvolution:
    KrDates2 = [dt.datetime.fromtimestamp(ts) for ts in KrPredictionUnixtimes]

#ax2.hlines(0, min(Dates2), max(Dates2), color='deeppink')
#ax2.fill_between(
#                         Dates2,
#                         PredictedELifeLowerErrors,
#                         PredictedELifeUpperErrors,
#                         color='b',
#                         alpha=0.5,
#                        )

XLimLow = dt.datetime.fromtimestamp(FirstPointUnixTime)
#XLimLow = dt.datetime.fromtimestamp(ScienceRunUnixtimes['SR1'][0] - 5*24.*3600)
#XLimLow = datetime.datetime(2017, 3, 10)
#XLimLow = datetime.datetime(2017, 5, 10)
#XLimLow = dt.datetime.fromtimestamp(1485802500)
#XLimUp = dt.datetime.fromtimestamp(LastPointUnixtime+DaysAfterLastPoint*3600.*24.)
XLimUp = datetime.datetime(2017, 9, 18)



YLs = [-20, -20]
YUs = [20, 20]
if ShowResiduals:
    for ScienceRun in ScienceRunUnixtimes.keys():
        if FillScienceRuns:
            Xs = [
                      dt.datetime.fromtimestamp(ScienceRunUnixtimes[ScienceRun][0]),
                      dt.datetime.fromtimestamp(ScienceRunUnixtimes[ScienceRun][1])
                     ]
            ax2.fill_between(Xs, YLs, YUs, color='coral', alpha=0.5)

        sr_time_end = min(time.mktime(XLimUp.timetuple()), ScienceRunUnixtimes[ScienceRun][1] )

        if DisplayScienceRun:
            ax2.text(
    #                    dt.datetime.fromtimestamp(0.5*( ScienceRunUnixtimes[ScienceRun][0] +
    #                                                    ScienceRunUnixtimes[ScienceRun][1] )),
    #                    0.5* (dt.datetime.fromtimestamp(ScienceRunUnixtimes[ScienceRun][1]) + dt.datetime.fromtimestamp(sr_time_end)),
                        dt.datetime.fromtimestamp(0.5*(ScienceRunUnixtimes[ScienceRun][0] + sr_time_end)),
                        10,
                        'Science run %s' %ScienceRun[2:],
                        color='k',
                        horizontalalignment='center',
                        size=28.
                        #size=35.,
                        )
            ax2.text(
    #                    dt.datetime.fromtimestamp(0.5*( ScienceRunUnixtimes[ScienceRun][0] +
    #                                                    ScienceRunUnixtimes[ScienceRun][1] )),
                        dt.datetime.fromtimestamp(0.5*(ScienceRunUnixtimes[ScienceRun][0] + sr_time_end)),
                        -10,
                        "RMS = "+str('%.2f' % RMSBias[ScienceRun])+"$\%$",
                        color='k',
                        horizontalalignment='center',
                        size=28.
                        #size=35.,
                        )


    ax2.errorbar(Dates, ELifeValueDeviations, xerr=[DateErrorLowers,DateErrorUppers],
                    yerr=[ELifeValueDeviationErrors,ELifeValueDeviationErrors], fmt='o', color='k')
    ax2.errorbar(PoRnDates, PoRnELifeValueDeviations,  xerr=[PoRnDateErrorLowers,PoRnDateErrorUppers],
                    yerr=[PoRnELifeValueDeviationErrors,PoRnELifeValueDeviationErrors], fmt='o', color='deeppink')
    ax2.errorbar(Rn222Dates, Rn222ELifeValueDeviations,  xerr=[Rn222DateErrorLowers,Rn222DateErrorUppers],
                    yerr=[Rn222ELifeValueDeviationErrors,Rn222ELifeValueDeviationErrors], fmt='o', color='gold',
                    marker='v', markersize=8)

    ax2.errorbar(Rn220Dates, Rn220ELifeValueDeviations,  xerr=[Rn220DateErrorLowers,Rn220DateErrorUppers],
                    yerr=[Rn220ELifeValueDeviationErrors,Rn220ELifeValueDeviationErrors], fmt='o', color='orangered',
                    marker='v', markersize=8)

#    ax2.set_xlim([XLimLow, XLimUp])
#    ax2.set_ylim([-20, 20])
    ax2.set_xlabel('Date', fontsize=30)
    ax2.set_ylabel('Residuals [%]', fontsize=30)
#    ax2.set_ylim([-15, 15])
#    ax2.set_ylim([-5, 5])
    ax2.set_ylim([-7, 7])
#fig = plt.figure(figsize=(25, 25))
#gs1 = gridspec.GridSpec(10,1)
#if ShowResiduals:
#ax = plt.subplot(gs1[0:2,:])
#ax2 = plt.subplot(gs1[2:3,:], sharex=ax)
#axs = [plt.subplot(gs1[3:4,:], sharex=ax)]
#axs.append(plt.subplot(gs1[4:5,:], sharex=ax))
#axs.append(plt.subplot(gs1[5:6,:], sharex=ax))
#axs.append(plt.subplot(gs1[6:7,:], sharex=ax))
#axs.append(plt.subplot(gs1[7:8,:], sharex=ax))
#axs.append(plt.subplot(gs1[8:9,:], sharex=ax))
#axs.append(plt.subplot(gs1[9:,:], sharex=ax))


#xfmt = md.DateFormatter('%Y-%m-%d')
#ax.xaxis.set_major_formatter(xfmt)

i = 0

#for sc_var in VariablesToPlot.keys():
for sc_var in d.keys():
    if sc_var not in VariablesToPlot:
        continue
    sc_datetimes = [datetime.datetime.fromtimestamp(u) for u in d[sc_var]['unixtimes']]
    values = d[sc_var]['values']
    axs[i].plot(sc_datetimes, values, label=VariablesToPlot[sc_var])
    axs[i].grid(linestyle='-', alpha=0.3)
    axs[i].axhline(np.mean(values), color='r', linewidth=3) 
    axs[i].axvline(datetime.datetime(2017, 7, 12), linewidth=3, color='r')
    axs[i].set_ylim(yLimsToPlot[sc_var])
    axs[i].tick_params(axis='x', labelsize=30)
    axs[i].tick_params(axis='y', labelsize=30)
    leg = axs[i].legend(loc='upper left', prop={'size':30})
    leg.get_frame().set_alpha(0)
    i += 1

ax.grid(linestyle='-', alpha=0.3)
ax2.grid(linestyle='-', alpha=0.3)
ax.axvline(datetime.datetime(2017, 7, 12), linewidth=3, color='r')
ax2.axvline(datetime.datetime(2017, 7, 12), linewidth=3, color='r')

#plt.show()

#ax.grid(True)
ax.set_xlim([XLimLow, XLimUp])
#ax.set_ylim([0, 700])
#ax.set_ylim([450, 680])
ax.set_ylim([250, 700])

handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
#ax.legend(by_label.values(), by_label.keys(), loc = 'lower right',prop={'size':20}, ncol=3)
#ax.legend(by_label.values(), by_label.keys(), loc = 'best',prop={'size':30}, ncol=2)

#ax.legend(loc = 'lower right',prop={'size':20})
ax.set_xlabel('Date', fontsize=30)
ax.set_ylabel('Electron lifetime $[\\mu s]$', fontsize=30)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

ax2.tick_params(axis='x', labelsize=30)
ax2.tick_params(axis='y', labelsize=30)

gs1.update(hspace=0)

ax.set_xlim([XLimLow, XLimUp])

fig.autofmt_xdate()

#plt.savefig(FigureSaveName+".png", format='png')
#plt.savefig(FigureSaveName+".pdf", format='pdf')

plt.show()
