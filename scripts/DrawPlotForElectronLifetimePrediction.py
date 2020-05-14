import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import sys
import glob
import os
#os.environ['MPLCONFIGDIR'] = '/project/lgrandi/xenon1t/mplconfigs/'

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
#plt.style.use('SR0')
#plt.rc('text', usetex=True)
#plt.rc('font', family='sans-serif')

#params = {
#    'backend': 'Agg',
#    # colormap
#    'image.cmap' : 'viridis',
#    # figure
#    'figure.figsize' : (9, 6),
#    'font.size' : 32,
#    'font.family' : 'serif',
#    'font.serif' : ['Times'],
#    # axes
#    'axes.titlesize' : 42,
#    'axes.labelsize' : 32,
#    'axes.linewidth' : 2,
#    # ticks
#    'xtick.labelsize' : 24,
#    'ytick.labelsize' : 24,
#    'xtick.major.size' : 16,
#    'xtick.minor.size' : 8,
#    'ytick.major.size' : 16,
#    'ytick.minor.size' : 8,
#    'xtick.major.width' : 2,
#    'xtick.minor.width' : 2,
#    'ytick.major.width' : 2,
#    'ytick.minor.width' : 2,
#    'xtick.direction' : 'in',
#    'ytick.direction' : 'in',
#    # markers
#    'lines.markersize' : 12,
#    'lines.markeredgewidth' : 3,
#    'errorbar.capsize' : 0,
#    'lines.linewidth' : 6,
##    'lines.linestyle' : None,
##    'lines.marker' : None,
#    'savefig.bbox' : 'tight',
#    'legend.fontsize' : 30,
#    #'legend.fontsize': 18,
#    #'figure.figsize': (15, 5),
#    #'axes.labelsize': 18,
#    #'axes.titlesize':18,
#    #'xtick.labelsize':14,
#    #'ytick.labelsize':14
#    'axes.labelsize': 24,
#    'axes.titlesize':24,
#    'xtick.labelsize':18,
#    'ytick.labelsize':18,
#    'mathtext.fontset': 'dejavuserif'
#}

#plt.rcParams.update(params)
#plt.rc('text', usetex=True)

import datetime as dt
import time
import pickle
from collections import OrderedDict
from MCMC_Tools import *
import FormPars
import CommonTools

if len(sys.argv)<2:
    print("======== Syntax: =======")
    print("python DrawPlotForElectronLifetimePrediction.py .....")
    print("< prediction txt file> ")
    print("< days to show after last data point >")
    print("<save fig name (rel.)>")
    exit()

PredictionFile = sys.argv[1]
DaysAfterLastPoint = float(sys.argv[2])
FigureSaveName = sys.argv[3]

ScienceRunUnixtimes  = FormPars.GetScienceRunUnixtimes()

Xe40kevELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithXe40keV.txt'
Xe129mELifeDataFile  = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithXe129m.txt'
Xe131mELifeDataFile  = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithXe131m.txt'
ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeData_SingleScatter_ForV2Fitting.txt'
Rn220ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn220_1day_new_wobad.txt'
Rn222ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithRn222_1day_new_wobad.txt'
Po218ELifeDataFile   = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithPo218_1day_new_wobad.txt'
PoRnELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/old/pre_time_dependent_3d_fdc/ElectronLifetimeDataWithPoRn_mod.txt'
Kr83ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/Lifetimes/ElectronLifetimeDataWithKr83.txt'
PathToHaxDatasets    = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/hax/'
PathToFigure    = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/MCMC_Results/Figures/'

PlotS1S2  = True
PlotXe40  = False
PlotKr83  = True
PlotXe129 = False
PlotXe131 = False
PlotPoRn  = True
PlotPo218 = True
PlotRn222 = True
PlotRn220 = True

BlessedPlot        = False
ShowResiduals      = True
FillScienceRuns    = False
LineScienceRuns    = False
DisplayScienceRun  = False
SayPreliminary     = False
ShowCalibrations   = True
UpdateCalibrations = True
PlotOperations     = True
PrintOpText        = True
DrawOpLines        = True
ShowLegend         = True

PlotRnEvolution    = True
PlotKrEvolution    = False
RotateDates = True
XLimStart = 'beginning'
#XLimStart = 'sr0_ambe'
#XLimStart = 'sr0_ambe'
#XLimStart = 'SR0'
#XLimStart = 'SR1'
#XLimLow = dt.datetime(2018, 1, 1)
XLimLow = None
XLimEnd = 'SR0'
#XLimEnd = 'end'
#XLimUp = datetime.datetime(2016, 10, 31)
#XLimUp = datetime.datetime(2018, 6, 20)
XLimUp = None
YLimLow = 0
YLimUp = 1100
YErrLimLow = -7.5
YErrLimUp = 7.5
#XLimUp = datetime.datetime(2016, 10, 31)

if XLimStart == 'beginning':
    YLimLow = 0
if XLimStart == 'SR0':
    YLimLow = 350
    YLimUp = 600
if XLimStart == 'SR1':
    YLimLow = 350
if XLimEnd == 'SR1':
    YLimUp = 600

FigEnding = ''
if not PlotRnEvolution:
    FigEnding = '_kronly'
    YErrLimLow = -3.
    YErrLimUp  = 3.

ResidEnding = ''
if ShowResiduals:
    ResidEnding = '_residuals'

ResidualPercentileTextColor = 'seagreen'
if PlotRnEvolution:
    ResidualPercentileTextColor = 'b'

PathContents         = PredictionFile.split('/')
PathToKrFolder       = '/'.join(PathContents[:-1]) + '/Kr83m/'
Kr83PredictionFile   = PathToKrFolder + 'Kr_' + PathContents[-1]
#Kr83PredictionFile   = 'MCMC_Results/TXTs/Kr83m/Kr_Prediction_180104sr0onlys_updated_correction_180108.txt'

if not os.path.exists(Kr83PredictionFile) and PlotKrEvolution:
    FileList           = PathToKrFolder + '*'
    SortedFiles        = sorted(glob.iglob(FileList), key=os.path.getctime, reverse=True)
    Kr83PredictionFile = SortedFiles[0]



#######################################
### Initialize hax.runs.datasets
#######################################
if ShowCalibrations:
    SourceRuns = GetCalibrationTimes(
                    PathToHaxDatasets,
                    SourcesToUse=['kr83m', 'rn220', 'ambe', 'neutron_generator'],
                    UpdateCalibrations=UpdateCalibrations
                    )
    CalibrationColors = {
                            'ambe': 'c',
                            'kr83m': 'r',
                            'rn220': 'm',
                            'neutron_generator': 'b',
                            'led': '#515A5A'
                            }

#######################################
### Get elife data
#######################################
(   UnixTimes,
    UnixTimeErrors,
    ELifeValues,
    ELifeValueErrors) = LoadRawFitData('SingleScatter', PathToFile=ELifeDataFile)

(   PoRnUnixtimes,
    PoRnUnixtimeErrors,
    PoRnELifeValues,
    PoRnELifeValueErrors) = LoadRawFitData('PoRn', PathToFile=PoRnELifeDataFile)

(   Rn222Unixtimes,
    Rn222UnixtimeErrors,
    Rn222ELifeValues,
    Rn222ELifeValueErrors) = LoadRawFitData('Rn222', PathToFile=Rn222ELifeDataFile)

#Fields = [0.0583245443043, 0.0787113537639, 0.117382423495, 0.162989973122, 0.253660532825, 0.34227580980800004, 0.436223335797, 0.588573683763, 0.7941034911479999, 1.0868133019500001, 1.38497072037, 1.81605663549, 2.41539631008, 3.44978922077, 3.9223289353700004, 5.8466333187299995, 7.24083005596]
#AttachingRates = [173839448697.0, 166490987263.0, 154902864208.0, 144105038741.0, 124672767516.0, 112649545428.0, 103268822098.0, 90632673738.3, 78393350018.2, 67808393527.7, 59505871959.5, 52222275531.1, 44516323097.2, 35805292980.3, 33787097454.0, 26016257301.4, 22829772653.1]
#AttachingRateAsField = interp1d(Fields, AttachingRates, fill_value = 'extrapolate')

#d = [datetime.datetime.utcfromtimestamp(ui) for ui in Rn222Unixtimes]
#Rn222Unixtimes = np.asarray(Rn222Unixtimes)
#Rn222ELifeValues = np.asarray(Rn222ELifeValues)
#Rn222ELifeValueErrors = np.asarray(Rn222ELifeValueErrors)

#Rn222ELifeValues[Rn222Unixtimes < 1485432700] *= AttachingRateAsField(0.12)
#Rn222ELifeValues[Rn222Unixtimes > 1485432700] *= AttachingRateAsField(0.08)

#plt.errorbar(
#    d,
#    1./np.asarray(Rn222ELifeValues),
#    yerr=np.asarray(Rn222ELifeValueErrors)/np.asarray(Rn222ELifeValues)**2,
#)

#plt.show()

(   Po218Unixtimes,
    Po218UnixtimeErrors,
    Po218ELifeValues,
    Po218ELifeValueErrors) = LoadRawFitData('Po218', PathToFile=Po218ELifeDataFile)

(   Rn220Unixtimes,
    Rn220UnixtimeErrors,
    Rn220ELifeValues,
    Rn220ELifeValueErrors) = LoadRawFitData('Rn220', PathToFile=Rn220ELifeDataFile)


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




if PlotPoRn:
    CutID = 0
    for i, unixtime in enumerate(UnixTimes):
        if unixtime>PoRnUnixtimes[0]:
            CutID = i
            break

    UnixTimes = UnixTimes[:CutID]
    UnixTimeErrors = UnixTimeErrors[:CutID]
    ELifeValues = ELifeValues[:CutID]
    ELifeValueErrors = ELifeValueErrors[:CutID]

if PlotPo218 or PlotRn220 or PlotRn222:
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
                                                        PredictionFile+".txt",
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


(   PredictionUnixtimes,
    PredictedIgs,
    PredictedIgLowers,
    PredictedIgUppers,
    PredictedIgLowerErrors,
    PredictedIgUpperErrors) = LoadPredictions(
                                                        PredictionFile+"_Ig.txt",
                                                        LastPointUnixtime=LastPointUnixtime,
                                                        DaysAfterLastPoint=DaysAfterLastPoint
                                                        )

(   LowerFitIgUncertainty,
    UpperFitIgUncertainty) = GetPredictionUncertainties(
                                                            PredictionUnixtimes,
                                                            PredictedIgs,
                                                            PredictedIgLowerErrors,
                                                            PredictedIgUpperErrors
                                                            )
(   PredictionUnixtimes_chi2,
    Predictedchi2s,
    Predictedchi2Lowers,
    Predictedchi2Uppers,
    Predictedchi2LowerErrors,
    Predictedchi2UpperErrors) = LoadPredictions(
                                                        PredictionFile+"_chi2.txt",
                                                        LastPointUnixtime=LastPointUnixtime,
                                                        DaysAfterLastPoint=DaysAfterLastPoint
                                                        )

(   LowerFitchi2Uncertainty,
    UpperFitchi2Uncertainty) = GetPredictionUncertainties(
                                                            PredictionUnixtimes,
                                                            Predictedchi2s,
                                                            Predictedchi2LowerErrors,
                                                            Predictedchi2UpperErrors
                                                            )

(   PredictionUnixtimes_Rchi2,
    PredictedRchi2s,
    PredictedRchi2Lowers,
    PredictedRchi2Uppers,
    PredictedRchi2LowerErrors,
    PredictedRchi2UpperErrors) = LoadPredictions(
                                                        PredictionFile+"_Rchi2.txt",
                                                        LastPointUnixtime=LastPointUnixtime,
                                                        DaysAfterLastPoint=DaysAfterLastPoint
                                                        )

(   LowerFitRchi2Uncertainty,
    UpperFitRchi2Uncertainty) = GetPredictionUncertainties(
                                                            PredictionUnixtimes,
                                                            PredictedRchi2s,
                                                            PredictedRchi2LowerErrors,
                                                            PredictedRchi2UpperErrors
                                                            )



(   PredictionUnixtimes,
    PredictedIls,
    PredictedIlLowers,
    PredictedIlUppers,
    PredictedIlLowerErrors,
    PredictedIlUpperErrors) = LoadPredictions(
                                                        PredictionFile+"_Il.txt",
                                                        LastPointUnixtime=LastPointUnixtime,
                                                        DaysAfterLastPoint=DaysAfterLastPoint
                                                        )

(   LowerFitIlUncertainty,
    UpperFitIlUncertainty) = GetPredictionUncertainties(
                                                            PredictionUnixtimes,
                                                            PredictedIls,
                                                            PredictedIlLowerErrors,
                                                            PredictedIlUpperErrors
                                                            )

#(   PredictionUnixtimeChi2s,
#    PredictedChi2s,
#    PredictedChi2Lowers,
#    PredictedChi2Uppers,
#    PredictedChi2LowerErrors,
#    PredictedChi2UpperErrors) = LoadPredictions(
#                                                        PredictionFile+"_Chi2.txt",
#                                                        LastPointUnixtime=LastPointUnixtime,
#                                                        DaysAfterLastPoint=DaysAfterLastPoint
#                                                        )
#
#(   LowerFitChi2Uncertainty,
#    UpperFitChi2Uncertainty) = GetPredictionUncertainties(
#                                                            PredictionUnixtimes,
#                                                            PredictedChi2s,
#                                                            PredictedChi2LowerErrors,
#                                                            PredictedChi2UpperErrors
#                                                            )



Rn222PredictionInterpolator = interp1d(PredictionUnixtimes, Rn222PredictedELifes, bounds_error=False, fill_value=(PredictionUnixtimes[0], PredictionUnixtimes[-1]))

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


    (   Kr83ELifeValueDeviations,
        Kr83ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                                KrUnixtimes,
                                                                KrELifeValues,
                                                                KrELifeValueErrors,
                                                                Kr83PredictionInterpolator
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
    try:
        MeanBias[ScienceRun], RMSBias[ScienceRun] = GetBiases(TotalUnixtimes, TotalELifeDeviations,
                                                              StartUnixtime=ScienceRunUnixtimes[ScienceRun][0],
                                                              EndUnixtime=ScienceRunUnixtimes[ScienceRun][1])

        print(ScienceRun + 'bias:   %.2f,\t '%(MeanBias[ScienceRun]) + ScienceRun + ' RMS:   %.2f' %(RMSBias[ScienceRun]))
    except:
        pass


###################################
## Calculate the uncertainties in the first science run
###################################


###################################
## convert unixtimes to dates
###################################
Dates        = [dt.datetime.utcfromtimestamp(ts) for ts in UnixTimes]
PoRnDates    = [dt.datetime.utcfromtimestamp(ts) for ts in PoRnUnixtimes]
Po218Dates   = [dt.datetime.utcfromtimestamp(ts) for ts in Po218Unixtimes]
Rn220Dates   = [dt.datetime.utcfromtimestamp(ts) for ts in Rn220Unixtimes]
Rn222Dates   = [dt.datetime.utcfromtimestamp(ts) for ts in Rn222Unixtimes]
KrDates      = [dt.datetime.utcfromtimestamp(ts) for ts in KrUnixtimes]
Xe40kevDates = [dt.datetime.utcfromtimestamp(ts) for ts in Xe40kevUnixtimes]
Xe129mDates  = [dt.datetime.utcfromtimestamp(ts) for ts in Xe129mUnixtimes]
Xe131mDates  = [dt.datetime.utcfromtimestamp(ts) for ts in Xe131mUnixtimes]

DateErrorLowers = []
DateErrorUppers = []
for ts, ts_err in zip(UnixTimes, UnixTimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    DateErrorLowers.append( date_err_lower )
    DateErrorUppers.append( date_err_upper )

KrDateErrorLowers = []
KrDateErrorUppers = []
for ts, ts_err in zip(KrUnixtimes, KrUnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    KrDateErrorLowers.append( date_err_lower )
    KrDateErrorUppers.append( date_err_upper )

PoRnDateErrorLowers = []
PoRnDateErrorUppers = []
for ts, ts_err in zip(PoRnUnixtimes, PoRnUnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    PoRnDateErrorLowers.append( date_err_lower )
    PoRnDateErrorUppers.append( date_err_upper )

Po218DateErrorLowers = []
Po218DateErrorUppers = []
for ts, ts_err in zip(Po218Unixtimes, Po218UnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Po218DateErrorLowers.append( date_err_lower )
    Po218DateErrorUppers.append( date_err_upper )

Rn220DateErrorLowers = []
Rn220DateErrorUppers = []
for ts, ts_err in zip(Rn220Unixtimes, Rn220UnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Rn220DateErrorLowers.append( date_err_lower )
    Rn220DateErrorUppers.append( date_err_upper )

Rn222DateErrorLowers = []
Rn222DateErrorUppers = []
for ts, ts_err in zip(Rn222Unixtimes, Rn222UnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Rn222DateErrorLowers.append( date_err_lower )
    Rn222DateErrorUppers.append( date_err_upper )

Xe40kevDateErrorLowers = []
Xe40kevDateErrorUppers = []
for ts, ts_err in zip(Xe40kevUnixtimes, Xe40kevUnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Xe40kevDateErrorLowers.append( date_err_lower )
    Xe40kevDateErrorUppers.append( date_err_upper )

Xe129mDateErrorLowers = []
Xe129mDateErrorUppers = []
for ts, ts_err in zip(Xe129mUnixtimes, Xe129mUnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Xe129mDateErrorLowers.append( date_err_lower )
    Xe129mDateErrorUppers.append( date_err_upper )

Xe131mDateErrorLowers = []
Xe131mDateErrorUppers = []
for ts, ts_err in zip(Xe131mUnixtimes, Xe131mUnixtimeErrors):
    date = dt.datetime.utcfromtimestamp(ts)
    date_err_lower = date - dt.datetime.utcfromtimestamp(ts - ts_err)
    date_err_upper = dt.datetime.utcfromtimestamp(ts + ts_err) - date
    Xe131mDateErrorLowers.append( date_err_lower )
    Xe131mDateErrorUppers.append( date_err_upper )
#UnixTimes2 = np.asarray(UnixTimes2)
#PredictedELifes = np.asarray(PredictedELifes)
#PredictedELifeLowers = np.asarray(PredictedELifeLowers)
#PredictedELifeUppers = np.asarray(PredictedELifeUppers)
#Dates2 = [dt.datetime.utcfromtimestamp(ts) for ts in UnixTimes2[UnixTimes2 < 1484731512]]
Dates2 = [dt.datetime.utcfromtimestamp(ts) for ts in PredictionUnixtimes]
if PlotKrEvolution:
    Dates3 = [dt.datetime.utcfromtimestamp(ts) for ts in KrPredictionUnixtimes]


##############################
## Draw plot
##############################


fig = plt.figure(figsize=(48.0,40))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

gs1 = gridspec.GridSpec(6,1)
if ShowResiduals:
    ax = plt.subplot(gs1[0:2,:])
    ax2 = plt.subplot(gs1[2:3,:], sharex=ax)
    ax3 = plt.subplot(gs1[3:4,:], sharex=ax)
    ax4 = plt.subplot(gs1[4:5,:], sharex=ax)
    ax5 = plt.subplot(gs1[5:6,:], sharex=ax)
else:
    ax = plt.subplot(gs1[0:3,:])

#xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
if BlessedPlot:
    xfmt = md.DateFormatter('%b %Y')
else:
    xfmt = md.DateFormatter('%Y-%m-%d')
#xfmt = md.DateFormatter('%Y-%m-%d')
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)


if ShowCalibrations:
    for Source in SourceRuns.keys():
        if Source == 'ambe':
            SourceLabel = '$\\rm{^{241}AmBe}\,\, Cal.$'
        elif Source == 'neutron_generator':
            SourceLabel = '$\\rm{NG \,\,Cal.}$'
        elif Source == 'kr83m':
            SourceLabel = '$\\rm{^{83m}Kr\,\, Cal.}$'
        elif Source == 'rn220':
            SourceLabel = '$\\rm{^{220}Rn\,\, Cal.}$'
        elif Source == 'led':
            SourceLabel = '$\\rm{LED \,\, Cal.}$'
        for Run in SourceRuns[Source]:
            ax.fill_between([Run[0], Run[1]],
                                YLimLow,
                                YLimUp,
                                color=CalibrationColors[Source],
                                alpha=0.1,
                                label=SourceLabel
                                )
            if ShowResiduals:
                ax2.fill_between([Run[0], Run[1]],
                                YErrLimLow,
                                YErrLimUp,
                                color=CalibrationColors[Source],
                                alpha=0.1
                                )
            ax3.fill_between([Run[0], Run[1]],
                                        0,
                                        1e3,
                                        color=CalibrationColors[Source],
                                        alpha=0.1
                                        )
            ax4.fill_between([Run[0], Run[1]],
                                        0,
                                        1e4,
                                        color=CalibrationColors[Source],
                                        alpha=0.1
                                        )
            ax5.fill_between([Run[0], Run[1]],
                                        0,
                                        10,
                                        color=CalibrationColors[Source],
                                        alpha=0.1
                                        )



if SayPreliminary:
    if XLimStart == 'beginning':
        XToPlot = dt.datetime(2016, 5, 23)
        YToPlot = 630. 
        Rotation = 45
    elif XLimStart == 'sr0_ambe':
        XToPlot = dt.datetime(2017, 3, 1)
        YToPlot = 520.
        Rotation= 30
    ax.text( 
                    XToPlot, 
                    YToPlot, 
                    'PRELIMINARY',
                    color='k',
                    size=72.,
                    rotation=Rotation,
                    alpha=0.2,
                    fontweight='heavy'
                    )




CathodeVoltages = FormPars.GetCathodeVoltages()

# plot times when voltage is not 0 kV, otherwise fill
for CathodeVoltage in CathodeVoltages:
    if CathodeVoltage[1][0] == 0 and CathodeVoltage[1][1] == 0:
#        ax.fill_between(Dates2, 0, 650, color='y', alpha=0.5, label=r'$V_{C} = 0$ kV')
        continue

    Dates2 = [dt.datetime.utcfromtimestamp(ts) for ts in PredictionUnixtimes
                if(ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

    Rn222ELifesToPlot       = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifes)
                                if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]
    Rn222ELifesLowToPlot    = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifeLowers)
                                if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]
    Rn222ELifesUpToPlot     = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222PredictedELifeUppers)
                                if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

    if PlotKrEvolution:
        KrUnixtimes2 = [ts for ts in KrPredictionUnixtimes
                    if(ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

        KrDates2 = [dt.datetime.utcfromtimestamp(ts) for ts in KrPredictionUnixtimes
                    if(ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

        Kr83ELifesToPlot       = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifes)
                                    if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]
        Kr83ELifesLowToPlot    = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifeLowers)
                                    if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]
        Kr83ELifesUpToPlot     = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83PredictedELifeUppers)
                                    if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

#LowerFitUncertainty
#UpperFitUncertainty

    if PlotKrEvolution:
        Kr83ELifesLowErrToPlot = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83LowerFitUncertainty)
                        if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

        Kr83ELifesUpErrToPlot = [ELife for ts,ELife in zip(KrPredictionUnixtimes,Kr83UpperFitUncertainty)
                        if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

    Rn222ELifesLowErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222LowerFitUncertainty)
                    if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

    Rn222ELifesUpErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,Rn222UpperFitUncertainty)
                    if (ts >= CathodeVoltage[0][0] and ts <= CathodeVoltage[0][1])]

    if PlotRnEvolution:
        ax.plot(
                    Dates2,
                    Rn222ELifesToPlot,
                    linewidth=6.,
                    color = 'r',
                    label='$\\rm{Best-fit\, trend\,\, (^{222}Rn)}$',
                   )
        ax.fill_between(
                                 Dates2,
                                 Rn222ELifesLowToPlot,
                                 Rn222ELifesUpToPlot,
                                 facecolor='b',
                                 edgecolor='none',
                                 label='$\\rm{68\% \,\, credible \,\, region\,\, (^{222}Rn)}$',
                                 alpha=0.5,
                                )

        Dates3 = [dt.datetime.utcfromtimestamp(ts) for ts in PredictionUnixtimes]

        ax3.plot(
                    Dates3,
                    PredictedIgs,
                    linewidth=6.,
                    color = 'r',
                   )
        ax3.fill_between(
                                 Dates3,
                                 PredictedIgLowers,
                                 PredictedIgUppers,
                                 facecolor='b',
                                 edgecolor='none',
                                 alpha=0.5,
                                )
        ax3.set_yscale("log", nonposy='clip')

        ax4.plot(
                    Dates3,
                    PredictedIls,
                    linewidth=6.,
                    color = 'r',
                   )
        ax4.fill_between(
                                 Dates3,
                                 PredictedIlLowers,
                                 PredictedIlUppers,
                                 facecolor='b',
                                 edgecolor='none',
                                 alpha=0.5,
                                )
        ax4.set_yscale("log", nonposy='clip')

        Dates4 = [dt.datetime.utcfromtimestamp(ts) for ts in PredictionUnixtimes_chi2]
        #ax5.plot(
        #            Dates4,
        #            Predictedchi2s,
        #            linewidth=6.,
        #            color = 'r',
        #           )
        #ax5.fill_between(
        #                         Dates4,
        #                         Predictedchi2Lowers,
        #                         Predictedchi2Uppers,
        #                         facecolor='b',
        #                         edgecolor='none',
        #                         alpha=0.5,
        #                        )

        #ax5.set_yscale("log", nonposy='clip')
        #ax5.grid()

        ax5.plot(
                    Dates4,
                    PredictedRchi2s,
                    linewidth=6.,
                    color = 'r',
                   )
        ax5.fill_between(
                                 Dates4,
                                 PredictedRchi2Lowers,
                                 PredictedRchi2Uppers,
                                 facecolor='b',
                                 edgecolor='none',
                                 alpha=0.5,
                                )
        ax5.grid()



#        ax5.plot(
#                    Dates4,
#                    PredictedChi2s,
#                    linewidth=6.,
#                    color = 'r',
#                   )
#
    if PlotKrEvolution:
        ax.plot(
                    KrDates2,
                    Kr83ELifesToPlot,
                    linewidth=6.,
                    color = 'g',
#                   color = 'gold',
                    label='$\\rm{Best-fit\, trend\,\, (^{83m}Kr)}$',
                   )
        if not BlessedPlot or not PlotRnEvolution:
            try:
              ax.fill_between(
                                       KrDates2,
                                       UnivariateSpline(KrUnixtimes2, Kr83ELifesLowToPlot, w=1000000./np.asarray(Kr83ELifesLowToPlot)**2)(KrUnixtimes2),
                                       UnivariateSpline(KrUnixtimes2, Kr83ELifesUpToPlot, w=1000000./np.asarray(Kr83ELifesUpToPlot)**2)(KrUnixtimes2),
                                       facecolor='seagreen',
                                       edgecolor='none',
                                       label='$\\rm{68 \% \,\, credible \,\, region\,\, (^{83m}Kr)}$',
                                       alpha=0.4,
                                      )
            except:
                pass
    if ShowResiduals and PlotRnEvolution:
        try:
          ax2.hlines(0, min(Dates2), max(Dates2), color='red', linewidth=6)
          ax2.fill_between(
                                   Dates2,
                                   Rn222ELifesLowErrToPlot,
                                   Rn222ELifesUpErrToPlot,
                                   facecolor='b',
                                   edgecolor='none',
                                   alpha=0.5,
                                  )
        except:
            pass

    elif ShowResiduals and PlotKrEvolution:
        try:
          ax2.hlines(0, min(Dates2), max(Dates2), color='g', linewidth=6)
          ax2.fill_between(
                                   KrDates2,
                                   Kr83ELifesLowErrToPlot,
                                   Kr83ELifesUpErrToPlot,
                                   facecolor='seagreen',
                                   edgecolor='none',
                                   alpha=0.4,
                                  )
        except:
            pass



if PlotS1S2:
    ax.errorbar(Dates, ELifeValues, xerr=[DateErrorLowers,DateErrorUppers],
                yerr=[ELifeValueErrors,ELifeValueErrors], fmt='o', color='k',
                label="S2/S1 method", elinewidth=2)

if PlotXe40:
    ax.errorbar(Xe40kevDates, Xe40kevELifeValues,  xerr = [Xe40kevDateErrorLowers,Xe40kevDateErrorUppers],
                yerr=[Xe40kevELifeValueErrors,Xe40kevELifeValueErrors], fmt='o', color='deepskyblue',
                marker='*', markersize=10, label='$\\rm{^{129}Xe}\,\,\, 39.6\,\, keV$', elinewidth=2)

if PlotKr83:
    ax.errorbar(KrDates, KrELifeValues, xerr=[KrDateErrorLowers, KrDateErrorUppers],
                yerr = [KrELifeValueErrors, KrELifeValueErrors],
                fmt = 'o', color = 'k', label ='$\\rm{^{83m}Kr}\,\,\, 41.6\,\, keV$', markersize=8, elinewidth=2)
if PlotXe129:
    ax.errorbar(Xe129mDates, Xe129mELifeValues,  xerr = [Xe129mDateErrorLowers,Xe129mDateErrorUppers],
                yerr=[Xe129mELifeValueErrors,Xe129mELifeValueErrors], fmt='o', color='darkmagenta',
                marker='*', markersize=10, label='$\\rm{^{131m}Xe}\,\,\, 163.9\,\, keV$', elinewidth=2)
if PlotXe131:
    ax.errorbar(Xe131mDates, Xe131mELifeValues,  xerr = [Xe131mDateErrorLowers,Xe131mDateErrorUppers],
                yerr=[Xe131mELifeValueErrors,Xe131mELifeValueErrors], fmt='o', color='darkorange',
                marker='*', markersize=10, label='$\\rm{^{129m}Xe}\,\,\, 236.1\,\, keV$', elinewidth=2)
if PlotPoRn:
    ax.errorbar(PoRnDates, PoRnELifeValues,  xerr = [PoRnDateErrorLowers,PoRnDateErrorUppers],
                yerr=[PoRnELifeValueErrors,PoRnELifeValueErrors], fmt='o', color='pink',
                marker='^', label='$\\rm{^{218}Po}/^{222}Rn}$', elinewidth=2)
if PlotRn220:
    ax.errorbar(Rn220Dates, Rn220ELifeValues,  xerr = [Rn220DateErrorLowers,Rn220DateErrorUppers],
                yerr=[Rn220ELifeValueErrors,Rn220ELifeValueErrors], fmt='o', color='green',
                marker='>', markersize=8, label='$\\rm{^{212}Bi}\,\,\, 6.207\,\, MeV$', elinewidth=2)

if PlotPo218:
    ax.errorbar(Po218Dates, Po218ELifeValues,  xerr = [Po218DateErrorLowers,Po218DateErrorUppers],
                yerr=[Po218ELifeValueErrors,Po218ELifeValueErrors], fmt='o', color='c',
                marker='v', markersize=8, label='$\\rm{^{218}Po}\,\,\, 6.114\,\, MeV$', elinewidth=2)
if PlotRn222:
    ax.errorbar(Rn222Dates, Rn222ELifeValues,  xerr = [Rn222DateErrorLowers,Rn222DateErrorUppers],
                yerr=[Rn222ELifeValueErrors,Rn222ELifeValueErrors], fmt='o', color='gold',
                marker='^', markersize=8, label='$\\rm{^{222}Rn}\,\,\, 5.590\,\, MeV$', elinewidth=2)

#TotalDates = [dt.datetime.utcfromtimestamp(ts) for ts in TotalUnixtimes]
#ax.errorbar(TotalDates, TotalELifeValues,
#            yerr=[TotalELifeValueErrors,TotalELifeValueErrors], fmt='o', color='r',
#            label="all")



# plot the vertical lines for system change
#ax.axvline( x=dt.datetime.utcfromtimestamp(1465937520), # first power outage
#                    ymin = 0,
#                    ymax = 700, 
#                    linestyle = "--",
#                    linewidth=3,
#                    color='k',
#                   )
#ax.axvline( x=dt.datetime.utcfromtimestamp(1468597800), # LN2 test. PTR1 warm-up
#                    ymin = 0,
#                    ymax = 700, 
#                    linestyle = "--",
#                    linewidth=3,
#                    color='k',
#                   )
#ax.axvline( x=dt.datetime.utcfromtimestamp(1484731512), # earthquake
#                    ymin = 0,
#                    ymax = 700, 
#                    linestyle = "--",
#                    linewidth=3,
#                    color='k',
#                   )
#

#XLimLow = dt.datetime.utcfromtimestamp(ScienceRunUnixtimes['SR1'][0] - 5*24.*3600)


# fill the region
Xs = [
          dt.datetime.utcfromtimestamp(1471880000),
          dt.datetime.utcfromtimestamp(1472800000)
         ]
YLs = [0, 0]
YUs = [700, 700]
#ax.fill_between(Xs, YLs, YUs, color='coral', alpha=0.7)
Xs = [
          dt.datetime.utcfromtimestamp(1475180000),
          dt.datetime.utcfromtimestamp(1475680000)
         ]
#ax.fill_between(Xs, YLs, YUs, color='m', alpha=0.3)

if FillScienceRuns:
    for ScienceRun in ScienceRunUnixtimes:
        Xs = [
                  dt.datetime.utcfromtimestamp(ScienceRunUnixtimes[ScienceRun][0]),
                  dt.datetime.utcfromtimestamp(ScienceRunUnixtimes[ScienceRun][1])
                 ]
        YLs = [0, 0]
        YUs = [700, 700]
        ax.fill_between(Xs, YLs, YUs, color='coral', alpha=0.5)


if XLimStart == 'beginning' and XLimLow is None:
    XLimLow = dt.datetime.utcfromtimestamp(FirstPointUnixTime)
if XLimStart == 'sr0_ambe' and XLimLow is None:
    XLimLow = SourceRuns['ambe'][0][0] - datetime.timedelta(weeks=1)
if XLimStart == 'SR0' and XLimLow is None:
    XLimLow = dt.datetime.utcfromtimestamp(ScienceRunUnixtimes['SR0'][0] - 5*24.*3600)
if XLimStart == 'SR1' and XLimLow is None:
    XLimLow = dt.datetime.utcfromtimestamp(ScienceRunUnixtimes['SR1'][0] - 5*24.*3600)
if XLimEnd == 'SR1' and XLimUp is None:
    XLimUp = dt.datetime.utcfromtimestamp(ScienceRunUnixtimes['SR1'][0] - 5*24.*3600)
elif XLimUp is None:
    XLimUp = dt.datetime.utcfromtimestamp(LastPointUnixtime+DaysAfterLastPoint*3600.*24.)
#XLimLow = dt.datetime.utcfromtimestamp(ScienceRunUnixtimes['SR0'][0] - 5*24.*3600)
#XLimLow = datetime.datetime(2017, 3, 10)
#XLimLow = datetime.datetime(2017, 5, 10)
#XLimLow = dt.datetime.utcfromtimestamp(1485802500)
#XLimUp = dt.datetime.utcfromtimestamp(LastPointUnixtime-20*3600.*24.)
#XLimUp = datetime.datetime(2017, 6, 10)
XLimLow = datetime.datetime(2016, 5, 10)
#XLimUp = datetime.datetime(2018, 2, 1)
#XLimUp = datetime.datetime(2018, 12, 1)
XLimUp = datetime.datetime(2018, 8, 1)
print(XLimLow)


if LineScienceRuns:
    for ScienceRun in ScienceRunUnixtimes.keys():
        TextOffsetStartRun = (YLimUp - YLimLow) * 1./70. + YLimLow
        TextOffsetEndRun = (YLimUp - YLimLow) * 1./70. + YLimLow
        VerticalAlignmentStartRun = 'bottom'
        VerticalAlignmentEndRun = 'bottom'

        if XLimStart == 'SR1' and ScienceRun == 'SR0':
            continue
        if XLimStart == 'sr0_ambe' and ScienceRun == 'SR0':
            VerticalAlignmentStartRun = 'top'
            VerticalAlignmentEndRun = 'top'
            TextOffsetStartRun = YLimUp - (YLimUp - YLimLow) * 1./70.
            TextOffsetEndRun = YLimUp - (YLimUp - YLimLow) * 1./70.
#        if XLimStart == 'sr0_ambe' and ScienceRun == 'SR1':
        if ScienceRun == 'SR1':
            VerticalAlignmentStartRun = 'top'
            TextOffsetStartRun = YLimUp - (YLimUp - YLimLow) * 1./70.
#        if XLimStart == 'sr0_ambe' and ScienceRun == 'SR1' and ShowLegend:
#            TextOffsetEndRun = 450

        StartUnixtime=ScienceRunUnixtimes[ScienceRun][0]
        EndUnixtime=ScienceRunUnixtimes[ScienceRun][1]
        if dt.datetime.utcfromtimestamp(StartUnixtime) > XLimLow and dt.datetime.utcfromtimestamp(StartUnixtime) < XLimUp:
        # start of science run
            ax.axvline( x=dt.datetime.utcfromtimestamp(StartUnixtime),
                                ymin = 0,
                                ymax = 1, 
                                linestyle = "--",
                                linewidth=3,
                                color='k',
                               )
            ax.text( 
                        dt.datetime.utcfromtimestamp(StartUnixtime + 2*24*3600), 
                        TextOffsetStartRun, 
                        'Start Science Run ' + ScienceRun.replace('SR', ''),
                        color='k',
                        size=32.,
                        rotation='vertical',
                        verticalalignment=VerticalAlignmentStartRun
                        )
        # end of science run
        if dt.datetime.utcfromtimestamp(EndUnixtime) > XLimLow and dt.datetime.utcfromtimestamp(EndUnixtime) < XLimUp:
            ax.axvline( x=dt.datetime.utcfromtimestamp(EndUnixtime),
                                ymin = 0,
                                ymax = 1, 
                                linestyle = "--",
                                linewidth=3,
                                color='k',
                               )
            ax.text( 
                        dt.datetime.utcfromtimestamp(EndUnixtime - 4*24*3600), 
                        TextOffsetEndRun, 
                        'End Science Run ' + ScienceRun.replace('SR', ''),
                        color='k',
                        size=32.,
                        rotation='vertical',
                        verticalalignment=VerticalAlignmentEndRun
                        )
        if ShowResiduals:
          if dt.datetime.utcfromtimestamp(StartUnixtime) > XLimLow and dt.datetime.utcfromtimestamp(StartUnixtime) < XLimUp:
                ax2.axvline( x=dt.datetime.utcfromtimestamp(StartUnixtime),
                                    ymin = 0,
                                    ymax = 1, 
                                    linestyle = "--",
                                    linewidth=3,
                                    color='k',
                                   )
          if dt.datetime.utcfromtimestamp(EndUnixtime) > XLimLow and dt.datetime.utcfromtimestamp(EndUnixtime) < XLimUp:
                ax2.axvline( x=dt.datetime.utcfromtimestamp(EndUnixtime),
                                    ymin = 0,
                                    ymax = 1, 
                                    linestyle = "--",
                                    linewidth=3,
                                    color='k',
                                   )
          if PlotOperations:
              Operations = FormPars.GetOperations(XLimStart, ShowLegend, PlotRnEvolution=PlotRnEvolution, ShowResiduals=ShowResiduals)
              for Operation,Info in Operations.items():
                  if Info['unixtime'] < time.mktime(XLimLow.timetuple()) and Info['draw']:
                      continue
                  if DrawOpLines:
                      ax2.axvline(
                          x=dt.datetime.utcfromtimestamp(Info['unixtime']),
                          ymin = 0,
                          ymax = 1, 
                          linestyle = Info.get('linestyle', '--'),
                          linewidth=Info.get('linewidth', 3),
                          color=Info['color'],
                      )



# plot operations text
if PlotOperations:
    Operations = FormPars.GetOperations(XLimStart, ShowLegend, PlotRnEvolution=PlotRnEvolution, ShowResiduals=ShowResiduals)
    for Operation,Info in Operations.items():
        if Info['unixtime'] < time.mktime(XLimLow.timetuple()) or not Info.get('draw', True):
            continue
        if DrawOpLines:
            ax.axvline(
                x=dt.datetime.utcfromtimestamp(Info['unixtime']),
                ymin = 0,
                ymax = 1, 
                linestyle = Info.get('linestyle', '--'),
                linewidth=Info.get('linewidth', 2),
                color=Info['color'],
                alpha=Info.get('alpha', 1),
            )
        if PrintOpText:
            vertical_alignment = Info.get('vertical_alignment', 'top')
            if Info['y'] == 'top':
                y = YLimUp - (YLimUp - YLimLow) * 1./70.
            elif Info['y'] == 'bottom':
                y = (YLimUp - YLimLow) * 1./70. + YLimLow
            else:
                y = Info['y']
            ax.text(
                dt.datetime.utcfromtimestamp(Info['unixtime'] + Info['print_text_after']*3600.*24),
                y,
                Info['text'],
                color=Info['color'],
                size=Info['size'],
                rotation=Info['rotation'],
                verticalalignment=vertical_alignment,
                alpha=Info.get('alpha', 1),
            )
#    ax.text( # Power outage
#                dt.datetime.utcfromtimestamp(1465937520+2.*3600.*24.), 
#                200., 
#                'PTR warm-up',
#                color='k',
#                size=22.,
#                rotation='vertical',
#                )
#    ax.text( # LN2 test, PTR warm up
#                dt.datetime.utcfromtimestamp(1468597800+2.*3600.*24.), 
#                450., 
#                'LN2 cooling test; PTR warm-up',
#                color='k',
#                size=22.,
#                rotation='vertical',
#                )
#    ax.text( # Earthquake @ 01/18/2017
#                dt.datetime.utcfromtimestamp(1484731512+2.*3600.*24.), 
#                450., 
#                'Earthquake @ 01/18/17',
#                color='k',
#                size=22.,
#                rotation='vertical',
#                )
#    ax.text( # Gas-only circulation
#                dt.datetime.utcfromtimestamp(1471880000-7.*3600.*24.), 
#                680., 
#                'Gas-only circulation',
#                color='coral',
#                size=22.,
#                #rotation='vertical',
#                )
#    ax.text(dt.datetime.utcfromtimestamp(1471880000), 580+20, "20 SLPM", color='coral', size=22.)
#    ax.text( # PUR upgrade
#                dt.datetime.utcfromtimestamp(1475180000-5.*3600.*24.), 
#                700., 
#                'PUR upgrade',
#                color='m',
#                size=22.,
#                alpha=0.5,
#                #rotation='vertical',
#                )
#    ax.text( # gate washing
#                dt.datetime.utcfromtimestamp(1496685600), 
#                700., 
#                'Gate Washing',
#                color='m',
#                size=22.,
#                alpha=0.5,
#                rotation='vertical',
#                )


# text the flow rate
#ax.text( dt.datetime.utcfromtimestamp(1464000000), 580+40., "$\sim$ 40 SLPM", size=20.,color='k')
#ax.text( dt.datetime.utcfromtimestamp(1466500000), 580+20, "$\sim$ 55 SLPM", size=20.,color='k')
#ax.text( dt.datetime.utcfromtimestamp(1469500000), 580+40, "45 - 50 SLPM", size=20.,color='k')
#ax.text( dt.datetime.utcfromtimestamp(1473500000), 580+40, "$\sim$ 40 SLPM", size=20.,color='k')
#ax.text( dt.datetime.utcfromtimestamp(1475700000), 580+20, "$\sim$ 54 SLPM", size=20.,color='k')


Dates2 = [dt.datetime.utcfromtimestamp(ts) for ts in PredictionUnixtimes]
if PlotKrEvolution:
    KrDates2 = [dt.datetime.utcfromtimestamp(ts) for ts in KrPredictionUnixtimes]

#ax2.hlines(0, min(Dates2), max(Dates2), color='deeppink')
#ax2.fill_between(
#                         Dates2,
#                         PredictedELifeLowerErrors,
#                         PredictedELifeUpperErrors,
#                         color='b',
#                         alpha=0.5,
#                        )


YLs = [-20, -20]
YUs = [20, 20]
if ShowResiduals:
    for ScienceRun in ScienceRunUnixtimes.keys():
        if FillScienceRuns:
            Xs = [
                      dt.datetime.utcfromtimestamp(ScienceRunUnixtimes[ScienceRun][0]),
                      dt.datetime.utcfromtimestamp(ScienceRunUnixtimes[ScienceRun][1])
                     ]
            ax2.fill_between(Xs, YLs, YUs, color='coral', alpha=0.5)

        sr_time_end = min(time.mktime(XLimUp.timetuple()), ScienceRunUnixtimes[ScienceRun][1] )


#        if BlessedPlot:
#            ax2.text(
#                        XLimLow + 0.8*(XLimUp - XLimLow),
#                        -5,
#                        '$\\rm{68 \% \,\, credible \,\, region\,\, (^{83m}Kr)}$',
#                        '$\\rm{68 \% \,\, credible \,\, region}$',
#                        '68% credible region',
#                        color=ResidualPercentileTextColor,
#                        horizontalalignment='center',
#                        size=30.,
#                        alpha=0.8,
#                        fontweight='heavy'
#                        )

        if DisplayScienceRun:
            ax2.text(
    #                    dt.datetime.utcfromtimestamp(0.5*( ScienceRunUnixtimes[ScienceRun][0] +
    #                                                    ScienceRunUnixtimes[ScienceRun][1] )),
    #                    0.5* (dt.datetime.utcfromtimestamp(ScienceRunUnixtimes[ScienceRun][1]) + dt.datetime.utcfromtimestamp(sr_time_end)),
                        dt.datetime.utcfromtimestamp(0.5*(ScienceRunUnixtimes[ScienceRun][0] + sr_time_end)),
                        10,
                        'Science run %s' %ScienceRun[2:],
                        color='k',
                        horizontalalignment='center',
                        size=28.
                        #size=35.,
                        )
            ax2.text(
    #                    dt.datetime.utcfromtimestamp(0.5*( ScienceRunUnixtimes[ScienceRun][0] +
    #                                                    ScienceRunUnixtimes[ScienceRun][1] )),
                        dt.datetime.utcfromtimestamp(0.5*(ScienceRunUnixtimes[ScienceRun][0] + sr_time_end)),
                        -10,
                        "RMS = "+str('%.2f' % RMSBias[ScienceRun])+"$\%$",
                        color='k',
                        horizontalalignment='center',
                        size=28.
                        #size=35.,
                        )

    if PlotS1S2:
        ax2.errorbar(Dates, ELifeValueDeviations, xerr=[DateErrorLowers,DateErrorUppers],
                        yerr=[ELifeValueDeviationErrors,ELifeValueDeviationErrors], fmt='o', color='k', elinewidth=2)

    if PlotPoRn:
        ax2.errorbar(PoRnDates, PoRnELifeValueDeviations,  xerr=[PoRnDateErrorLowers,PoRnDateErrorUppers],
                        yerr=[PoRnELifeValueDeviationErrors,PoRnELifeValueDeviationErrors], fmt='o', color='blue',
			elinewidth=2)
    if PlotPo218:
        ax2.errorbar(Po218Dates, Po218ELifeValueDeviations,  xerr=[Po218DateErrorLowers,Po218DateErrorUppers],
                        yerr=[Po218ELifeValueDeviationErrors,Po218ELifeValueDeviationErrors], fmt='o', color='c',
                        marker='v', markersize=8, elinewidth=2)

    if PlotRn222:
        ax2.errorbar(Rn222Dates, Rn222ELifeValueDeviations,  xerr=[Rn222DateErrorLowers,Rn222DateErrorUppers],
                        yerr=[Rn222ELifeValueDeviationErrors,Rn222ELifeValueDeviationErrors], fmt='o', color='gold',
                        marker='v', markersize=8, elinewidth=2)

    if PlotRn220:
        ax2.errorbar(Rn220Dates, Rn220ELifeValueDeviations,  xerr=[Rn220DateErrorLowers,Rn220DateErrorUppers],
                        yerr=[Rn220ELifeValueDeviationErrors,Rn220ELifeValueDeviationErrors], fmt='o',
                        color='green',
                        marker='v', markersize=8, elinewidth=2)

    if PlotKr83 and PlotKrEvolution:
        ax2.errorbar(KrDates, Kr83ELifeValueDeviations,  xerr=[KrDateErrorLowers,KrDateErrorUppers],
                        yerr=[Kr83ELifeValueDeviationErrors,Kr83ELifeValueDeviationErrors],
                        fmt='o', color='k', markersize=8, elinewidth=2)

#    ax2.set_xlim([XLimLow, XLimUp])
    ax2.set_xlabel('Date', fontsize=30)
#    ax2.set_ylabel('$\\rm{Residuals \,\, [\%]}$', fontsize=40, fontweight='heavy', labelpad=20)
    ax2.set_ylabel('$\\rm{Residuals \,\, [\%]}$', fontsize=30, labelpad=30)
    ax2.set_ylim([YErrLimLow, YErrLimUp])


#ax.grid(True)
ax.set_xlim([XLimLow, XLimUp])
ax.set_ylim([YLimLow, YLimUp])
#ax.set_ylim([450, 680])
#ax.set_ylim([450, 700])
#ax.set_ylim([300, 700])

#axtwin = ax.twinx()
#axtwint = ax.twinx()
#axtwins = ax.twinx()
#axtwinu = ax.twinx()
#axtwinv = ax.twinx()
#axtwinw = ax.twinx()
axtwiny = ax.twinx()
axtwinz = ax.twinx()
slowControl = pickle.load(open('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/SlowControl/PickleFiles/ImpurityUpdated_160501_to_180808.pkl', 'rb'))
#slowControl = pickle.load(open('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/SlowControl/PickleFiles/Coldfinger_160501_to_180717.pkl', 'rb'))
dsTPC = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['TPC_Monitor_Voltage']['unixtimes']]
dsCRY113 = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_TE113']['unixtimes']]
dsCRY104 = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_TE104']['unixtimes']]
dsPTR = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_R121P']['unixtimes']]

interp1 = interp1d(slowControl['PUR_FC201']['unixtimes'], slowControl['PUR_FC201']['values'], fill_value='extrapolate')
interp2 = interp1d(slowControl['PUR_FC202']['unixtimes'], slowControl['PUR_FC202']['values'], fill_value='extrapolate')
interp = interp1(slowControl['PUR_FC202']['unixtimes']) + interp2(slowControl['PUR_FC202']['unixtimes'])
dPUR = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['PUR_FC202']['unixtimes']]

ginterp1 = interp1d(slowControl['CRY_FCV101']['unixtimes'], slowControl['CRY_FCV101']['values'], fill_value='extrapolate')
ginterp2 = interp1d(slowControl['CRY_FCV102']['unixtimes'], slowControl['CRY_FCV102']['values'], fill_value='extrapolate')
ginterp3 = interp1d(slowControl['CRY_FCV103']['unixtimes'], slowControl['CRY_FCV103']['values'], fill_value='extrapolate')
ginterp = ginterp1(slowControl['CRY_FCV101']['unixtimes']) + ginterp2(slowControl['CRY_FCV101']['unixtimes']) + ginterp3(slowControl['CRY_FCV101']['unixtimes'])
dG = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_FCV101']['unixtimes']]
#dsTE131a = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_TE131a']['unixtimes']]
#dsTE121a = [datetime.datetime.utcfromtimestamp(ui) for ui in slowControl['CRY_TE121a']['unixtimes']]
#axtwinv.plot(
#    dsTE121a,
#    slowControl['CRY_TE121a']['values'],
#    color='b'
#)
#axtwinw.plot(
#    dsTE131a,
#    slowControl['CRY_TE131a']['values'],
#    color='k',
#    alpha=0.5
#)

#axtwin.plot(
#    dsTPC,
#    slowControl['TPC_Monitor_Voltage']['values'],
#    color='b',
#    alpha=0.5
#)
#axtwint.plot(
#    dsCRY113,
#    slowControl['CRY_TE113']['values'],
#    color='r',
#    alpha=0.5
#)
#axtwinu.plot(
#    dsPTR,
#    slowControl['CRY_R121P']['values'],
#    color='g',
#    alpha=0.5
#)
#axtwins.plot(
#    dsCRY104,
#    slowControl['CRY_TE104']['values'],
#    color='k',
#    alpha=0.5
#)
axtwiny.plot(
    dPUR,
    interp,
    color='fuchsia',
    alpha=0.5
)
axtwinz.plot(
    dG,
    ginterp,
    color='g',
    alpha=0.5
)


#print(slowControl['CRY_TE104']['values'][10000:10100])
#axtwin.set_xlim([XLimLow, XLimUp])
#axtwint.set_xlim([XLimLow, XLimUp])
#axtwins.set_xlim([XLimLow, XLimUp])
#axtwinu.set_xlim([XLimLow, XLimUp])
#axtwinv.set_xlim([XLimLow, XLimUp])
#axtwinw.set_xlim([XLimLow, XLimUp])
axtwiny.set_xlim([XLimLow, XLimUp])
axtwinz.set_xlim([XLimLow, XLimUp])
#axtwin.xaxis.set_major_formatter(xfmt)
#axtwint.xaxis.set_major_formatter(xfmt)
#axtwins.xaxis.set_major_formatter(xfmt)
#axtwin.set_ylim(0, 20)
#axtwint.set_ylim(12, 20)
#axtwins.set_ylim(-95.5, -97)
#axtwinu.set_ylim(12, 160)
#axtwinv.set_ylim(-100, -80)
#axtwinw.set_ylim(-100, -80)
#axtwiny.set_ylim(0, 120)
#plt.setp(axtwin.get_xticklabels(), visible=True)
#plt.setp(axtwint.get_xticklabels(), visible=True)
#plt.setp(axtwins.get_xticklabels(), visible=True)

#ordered_labels = [  '$\\rm{^{83m}Kr}\\,\\,\\, 41.6\\,\\, keV$',
#                    '$\\rm{Best-fit\\, trend\\,\\, (^{83m}Kr)}$',
#                    '$\\rm{68 \% \,\, credible \,\, region\,\, (^{83m}Kr)}$'
ordered_labels = []
if PlotKrEvolution:
    ordered_labels.extend(['$\\rm{Best-fit\\, trend\\,\\, (^{83m}Kr)}$'])
    if not PlotRnEvolution:
        ordered_labels.extend(['$\\rm{68 \% \,\, credible \,\, region\,\, (^{83m}Kr)}$'])

if PlotRnEvolution:
     ordered_labels.extend(['$\\rm{Best-fit\, trend\,\, (^{222}Rn)}$'])
     ordered_labels.extend(['$\\rm{68\% \,\, credible \,\, region\,\, (^{222}Rn)}$'])

#EmptyPatch = mpatches.Circle((-1, -1), color='white', label=' ', alpha=0.0)
#ax.add_patch(EmptyPatch)
#ordered_labels.extend([' '])

if PlotKr83:
    if PlotRnEvolution:
        ordered_labels.extend(['$\\rm{^{83m}Kr}\\,\\,\\, 41.6\\,\\, keV$'])
    else:
        ordered_labels.insert(0, '$\\rm{^{83m}Kr}\\,\\,\\, 41.6\\,\\, keV$')
if PlotRn220:
    ordered_labels.extend(['$\\rm{^{212}Bi}\,\,\, 6.207\,\, MeV$'])
if PlotPo218:
    ordered_labels.extend(['$\\rm{^{218}Po}\,\,\, 6.114\,\, MeV$'])
if PlotRn222:
    ordered_labels.extend(['$\\rm{^{222}Rn}\,\,\, 5.590\,\, MeV$'])

if ShowCalibrations:
    ordered_labels.extend([
                    '$\\rm{^{83m}Kr\,\, Cal.}$',
                    '$\\rm{^{220}Rn\,\, Cal.}$',
                    '$\\rm{^{241}AmBe}\,\, Cal.$',
                    '$\\rm{NG \,\,Cal.}$'
    ])

#ordered_labels = [  '$\\rm{^{83m}Kr}\\,\\,\\, 41.6\\,\\, keV$',
#                    '$\\rm{Best-fit\\, trend\\,\\, (^{83m}Kr)}$',

handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
by_label_ordered = OrderedDict()
for l in ordered_labels:
    by_label_ordered[l] = by_label[l]
for key,val in by_label.items():
    if key not in ordered_labels:
        by_label_ordered[key] = val

ax.set_xlim([XLimLow, XLimUp])
if ShowLegend:
#    ax.legend(by_label.values(), by_label.keys(), loc = 'lower right',prop={'size':20}, ncol=3)
    if PlotRnEvolution:
#        leg = ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = (0.2, 0.01) ,prop={'size':20}, ncol=3)
        leg = ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = 'bottom left' ,prop={'size':20}, ncol=3)
    else:
        leg = ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = (0.55, 0.01) ,prop={'size':20}, ncol=2)
#        leg = ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = (0.4, 0.01) ,prop={'size':30}, ncol=2)
#        leg = ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = 'lower left' ,prop={'size':30}, ncol=1)
#ax.legend(by_label.values(), by_label.keys(), loc = 'lower right',prop={'size':35}, ncol=1)
#ax.legend(by_label_ordered.values(), by_label_ordered.keys(), loc = 'best',prop={'size':25}, ncol=2)

#label = ax.yaxis.get_majorticklabels()[0]
#label.set_transform(label.get_transform() + transforms.ScaledTranslation(0, 0.5, fig.dpi_scale_trans))
#for label in ax.xaxis.get_majorticklabels():
#    label.set_transform(label.get_transform() + transforms.ScaledTranslation(0.5, 0, fig.dpi_scale_trans))

#ax.legend(loc = 'lower right',prop={'size':20})
ax.set_xlabel('$\\rm{Date}$', fontsize=40, fontweight='heavy')
#ax.set_ylabel('$\\rm{Electron \,\, lifetime \,\, [\\mu s]}$', fontsize=30, fontweight='heavy', labelpad=20)
ax.set_ylabel('$\\rm{Electron \,\, lifetime \,\, [\\mu s]}$', fontsize=40, labelpad=15)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

if ShowResiduals:
    plt.setp(ax.get_xticklabels(), visible=False)

ax.xaxis.set_tick_params(width=2, length=10)
ax.yaxis.set_tick_params(width=2, length=10)
if ShowResiduals:
  ax2.xaxis.set_tick_params(width=2, length=10)
  ax2.yaxis.set_tick_params(width=2, length=10)

  ax2.tick_params(axis='x', labelsize=30)
  ax2.tick_params(axis='y', labelsize=30)

  ax3.set_ylabel('$\\rm{Impurites \,\, in \,\, GXe [ppb]}$', fontsize=40, labelpad=15)
  ax3.tick_params(axis='x', labelsize=30)
  ax3.tick_params(axis='y', labelsize=30)

  ax4.set_ylabel('$\\rm{Impurites \,\, in \,\, LXe [ppb]}$', fontsize=40, labelpad=15)
  ax4.tick_params(axis='x', labelsize=30)
  ax4.tick_params(axis='y', labelsize=30)

  ax5.set_ylabel('$\\rm{Chi2}$', fontsize=40, labelpad=15)
  ax5.tick_params(axis='x', labelsize=30)
  ax5.tick_params(axis='y', labelsize=30)
  ax5.set_ylim([0, 5])

  #ax6.set_ylabel('$\\rm{Reduced Chi2}$', fontsize=40, labelpad=15)
  #ax6.tick_params(axis='x', labelsize=30)
  #ax6.tick_params(axis='y', labelsize=30)


  #ax5.set_ylabel('$\\rm{Impurites \,\, in \,\, Wall GXe [ppb]}$', fontsize=40, labelpad=15)
  #ax5.tick_params(axis='x', labelsize=30)
  #ax5.tick_params(axis='y', labelsize=30)


gs1.update(hspace=0)
plt.grid()

#ax.set_xlim([XLimLow, XLimUp])

if RotateDates:
    fig.autofmt_xdate()

plt.savefig(PathToFigure+FigureSaveName+'_'+XLimStart+ResidEnding+FigEnding+'.png', format='png')
plt.savefig(PathToFigure+FigureSaveName+'_'+XLimStart+ResidEnding+FigEnding+'.pdf', format='pdf')

#plt.show()
