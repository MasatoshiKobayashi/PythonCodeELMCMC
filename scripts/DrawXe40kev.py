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

Xe40kevELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeDataWithXe40keV.txt'
Xe129mELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeDataWithXe129m.txt'
Xe131mELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeDataWithXe131m.txt'
Rn222ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeWithRn222.txt'
Po218ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeWithPo218.txt'
Rn220ELifeDataFile = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/FitData/ElectronLifetimeWithRn220.txt'


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
    PredictedELifes,
    PredictedELifeLowers,
    PredictedELifeUppers,
    PredictedELifeLowerErrors,
    PredictedELifeUpperErrors) = LoadPredictions(
                                                    PredictionFile,
                                                    LastPointUnixtime,
                                                    DaysAfterLastPoint=DaysAfterLastPoint
                                                    )

PredictionInterpolator = interp1d(PredictionUnixtimes, PredictedELifes)

(   LowerFitUncertainty,
    UpperFitUncertainty) = GetPredictionUncertainties(
                                                        PredictionUnixtimes,
                                                        PredictedELifes,
                                                        PredictedELifeLowerErrors,
                                                        PredictedELifeUpperErrors
                                                        )





###################################
## Get the residual of the data points
###################################
(   ELifeValueDeviations,
    ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            UnixTimes,
                                                            ELifeValues,
                                                            ELifeValueErrors,
                                                            PredictionInterpolator
                                                            )

(   PoRnELifeValueDeviations,
    PoRnELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            PoRnUnixtimes,
                                                            PoRnELifeValues,
                                                            PoRnELifeValueErrors,
                                                            PredictionInterpolator
                                                            )

(   Rn222ELifeValueDeviations,
    Rn222ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Rn222Unixtimes,
                                                            Rn222ELifeValues,
                                                            Rn222ELifeValueErrors,
                                                            PredictionInterpolator
                                                            )

(   Po218ELifeValueDeviations,
    Po218ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Po218Unixtimes,
                                                            Po218ELifeValues,
                                                            Po218ELifeValueErrors,
                                                            PredictionInterpolator
                                                            )

(   Rn220ELifeValueDeviations,
    Rn220ELifeValueDeviationErrors) = GetLifetimeDeviations(
                                                            Rn220Unixtimes,
                                                            Rn220ELifeValues,
                                                            Rn220ELifeValueErrors,
                                                            PredictionInterpolator
                                                            )

TotalUnixtimes = UnixTimes + PoRnUnixtimes + Rn222Unixtimes
TotalELifeValues = ELifeValues + PoRnELifeValues + Rn222ELifeValues
TotalELifeValueErrors = ELifeValueErrors + PoRnELifeValueErrors + Rn222ELifeValueErrors
TotalELifeDeviations = ELifeValueDeviations + PoRnELifeValueDeviations + Rn222ELifeValueDeviations

TotalMeanBias, TotalRMSBias = GetBiases(TotalUnixtimes, TotalELifeDeviations)
SR0MeanBias, SR0RMSBias = GetBiases(TotalUnixtimes, TotalELifeDeviations)
SR1MeanBias, SR1RMSBias = GetBiases(TotalUnixtimes, TotalELifeDeviations)

print('Total bias: %.2f, Total RMS: %.2f' %(TotalMeanBias, TotalRMSBias))
print('SR0 bias:   %.2f, SR0 RMS:   %.2f' %(SR0MeanBias, SR0RMSBias))
print('SR1 bias:   %.2f, SR1 RMS:   %.2f' %(SR1MeanBias, SR1RMSBias))


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


##############################
## Draw plot
##############################
ScienceRunUnixtimes = FormPars.GetScienceRunUnixtimes()
ScienceRunStartUnixtime = 1479772800
ScienceRunEndUnixtime = 1484731512

XLimLow = dt.datetime.fromtimestamp(FirstPointUnixTime)
#XLimLow = dt.datetime.fromtimestamp(ScienceRunUnixtimes['SR1'][0])
#XLimLow = dt.datetime.fromtimestamp(1485802500)
XLimUp = dt.datetime.fromtimestamp(LastPointUnixtime+DaysAfterLastPoint*3600.*24.)


fig = plt.figure(figsize=(25.0, 16.0))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

gs1 = gridspec.GridSpec(3,1)
ax = plt.subplot(gs1[0:3,:])
#ax = plt.subplot(gs1[0:2,:])
#ax2 = plt.subplot(gs1[2:3,:])

xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)


CathodeVoltages = FormPars.GetCathodeVoltages()

# plot times when voltage is not 0 kV, otherwise fill
for CathodeVoltage in CathodeVoltages:
    Dates2 = [dt.datetime.fromtimestamp(ts) for ts in PredictionUnixtimes
                if(ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    if CathodeVoltage[1][0] == 0 and CathodeVoltage[1][1] == 0:
#        ax.fill_between(Dates2, 0, 650, color='y', alpha=0.5, label=r'$V_{C} = 0$ kV')
        continue

    ELifesToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,PredictedELifes)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    ELifesLowToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,PredictedELifeLowers)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    ELifesUpToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,PredictedELifeUppers)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    ELifesLowErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,PredictedELifeLowerErrors)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    ELifesUpErrToPlot = [ELife for ts,ELife in zip(PredictionUnixtimes,PredictedELifeUpperErrors)
                    if (ts >= CathodeVoltage[0][0] and ts < CathodeVoltage[0][1])]

    ax.plot(
                Dates2,
#                PredictedELifes,
                ELifesToPlot,
                linewidth=2.,
                color = 'r',
#                color = 'gold',
                alpha=0,
                label='Best-fit trend',
               )
    ax.fill_between(
                             Dates2,
                             ELifesLowToPlot,
                             ELifesUpToPlot,
#                             PredictedELifeLowers,
#                             PredictedELifeUppers,
                             color='b',
#                             color='c',
                             label=r'$\pm 1 \sigma$ C.L. region',
                             alpha=0.0,
                            )

    #ax2.hlines(0, min(Dates2), max(Dates2), color='deeppink')
    #ax2.fill_between(
    #                         Dates2,
    #                         ELifesLowErrToPlot,
    #                         ELifesUpErrToPlot,
    #                         color='b',
    #                         alpha=0.5,
    #                        )






dSourceDates = dict(
                    Kr83m = [
                            [1489144526, 1489244895],
                            [1491207468, 1491300817],
#                            [1492418717, 1492514479]
                            ],
                    Cs137 = [
                            [1491312264, 1491342302]
                            ],
                    Th228 = [
                            [1491379392, 1491406193]
                            ],
                    Rn220 = [
                            [1489402529, 1489671382]
                            ],
                    AmBe = [
                            [1489671382, 1490960046]
                            ]
                    )

dSourceColors = dict(
                    Kr83m = 'g',
                    Cs137 = 'orangered',
                    Th228 = 'purple',
                    Rn220 = 'dodgerblue',
                    AmBe = 'r'
                    )
yLims = ax.get_ylim()
TextHeight = 640
if True:
    for key,value in dSourceDates.items():
        for unixtimes in value:
                ax.fill_between([datetime.datetime.fromtimestamp(unixtimes[0]),
                                        datetime.datetime.fromtimestamp(unixtimes[1])],
                                        [0, 0], [yLims[1], yLims[1]],
                                        color=dSourceColors[key], alpha=0.3)
                plt.gca().text(datetime.datetime.fromtimestamp(unixtimes[0] + 3*3600),
                                TextHeight, key, color=dSourceColors[key],
                                rotation='vertical', fontsize=25)







#ax.errorbar(Dates, ELifeValues, xerr=[DateErrorLowers,DateErrorUppers],
#            yerr=[ELifeValueErrors,ELifeValueErrors], fmt='o', color='k',
#            label="electron lifetime data points (S2/S1 method)")

#ax.errorbar(PoRnDates, PoRnELifeValues,  xerr = [PoRnDateErrorLowers,PoRnDateErrorUppers],
#            yerr=[PoRnELifeValueErrors,PoRnELifeValueErrors], fmt='o', color='deeppink',
#            label="electron lifetime data points (from Po218/Rn222 analysis)")

ax.errorbar(Po218Dates, Po218ELifeValues,  xerr = [Po218DateErrorLowers,Po218DateErrorUppers],
            yerr=[Po218ELifeValueErrors,Po218ELifeValueErrors], fmt='o', color='c',
            label="Po218")

ax.errorbar(Rn222Dates, Rn222ELifeValues,  xerr = [Rn222DateErrorLowers,Rn222DateErrorUppers],
            yerr=[Rn222ELifeValueErrors,Rn222ELifeValueErrors], fmt='o', color='gold',
            label="Rn222")

ax.errorbar(KrDates, KrELifeValues, yerr = [KrELifeValueErrors, KrELifeValueErrors],
            fmt = 'o', color = 'g', label = "Kr83m")

ax.errorbar(Rn220Dates, Rn220ELifeValues,  xerr = [Rn220DateErrorLowers,Rn220DateErrorUppers],
            yerr=[Rn220ELifeValueErrors,Rn220ELifeValueErrors], fmt='o', color='orangered',
            label="Rn220")

ax.errorbar(Xe40kevDates, Xe40kevELifeValues,  xerr = [Xe40kevDateErrorLowers,Xe40kevDateErrorUppers],
            yerr=[Xe40kevELifeValueErrors,Xe40kevELifeValueErrors], fmt='o', color='darkmagenta',
            label="Xe 40 keV")

#ax.errorbar(Xe129mDates, Xe129mELifeValues,  xerr = [Xe129mDateErrorLowers,Xe129mDateErrorUppers],
#            yerr=[Xe129mELifeValueErrors,Xe129mELifeValueErrors], fmt='o', color='orangered',
#            label="electron lifetime data points (from Xe129m analysis)")

#ax.errorbar(Xe131mDates, Xe131mELifeValues,  xerr = [Xe131mDateErrorLowers,Xe131mDateErrorUppers],
#            yerr=[Xe131mELifeValueErrors,Xe131mELifeValueErrors], fmt='o', color='orangered',
#            label="electron lifetime data points (from Xe131m analysis)")


#TotalDates = [dt.datetime.fromtimestamp(ts) for ts in TotalUnixtimes]
#ax.errorbar(TotalDates, TotalELifeValues,
#            yerr=[TotalELifeValueErrors,TotalELifeValueErrors], fmt='o', color='r',
#            label="all")



# plot the vertical lines for system change
ax.axvline( x=dt.datetime.fromtimestamp(1465937520), # first power outage
                    ymin = 0,
                    ymax = 650, 
                    linestyle = "--",
                    linewidth=3,
                    color='k',
                   )
ax.axvline( x=dt.datetime.fromtimestamp(1468597800), # LN2 test. PTR1 warm-up
                    ymin = 0,
                    ymax = 650, 
                    linestyle = "--",
                    linewidth=3,
                    color='k',
                   )
ax.axvline( x=dt.datetime.fromtimestamp(1484731512), # earthquake
                    ymin = 0,
                    ymax = 650, 
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
YUs = [650, 650]
ax.fill_between(Xs, YLs, YUs, color='coral', alpha=0.7)
Xs = [
          dt.datetime.fromtimestamp(1475180000),
          dt.datetime.fromtimestamp(1475680000)
         ]
ax.fill_between(Xs, YLs, YUs, color='m', alpha=0.3)


Xs = [
          dt.datetime.fromtimestamp(ScienceRunStartUnixtime),
          dt.datetime.fromtimestamp(ScienceRunEndUnixtime)
         ]
YLs = [0, 0]
YUs = [650, 650]
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
            675., 
            'Gas-only circulation',
            color='coral',
            size=22.,
            #rotation='vertical',
            )
ax.text(dt.datetime.fromtimestamp(1471880000), 580+20, "20 SLPM", color='coral', size=22.)
ax.text( # PUR upgrade
            dt.datetime.fromtimestamp(1475180000-5.*3600.*24.), 
            660., 
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
#ax2.fill_between(Xs, YLs, YUs, color='coral', alpha=0.5)
#ax2.text(
#              dt.datetime.fromtimestamp(0.5*( ScienceRunStartUnixtime+ScienceRunEndUnixtime )),
#              15,
#              "Science run 0",
#              color='k',
#              horizontalalignment='center',
#              size=22.
#            size=35.,
#              )
#ax2.text(
#              dt.datetime.fromtimestamp(0.5*( ScienceRunStartUnixtime+ScienceRunEndUnixtime )),
#              -15,
#              "RMS = "+str('%.2f' % RMSBias)+"$\%$",
#              color='k',
#              horizontalalignment='center',
#              size=22.
#            size=35.,
#              )


#ax2.errorbar(Dates, ELifeValueDeviations, xerr=[DateErrorLowers,DateErrorUppers], yerr=[ELifeValueDeviationErrors,ELifeValueDeviationErrors], fmt='o', color='k')
#ax2.errorbar(PoRnDates, PoRnELifeValueDeviations,  xerr = [PoRnDateErrorLowers,PoRnDateErrorUppers], yerr=[PoRnELifeValueDeviationErrors,PoRnELifeValueDeviationErrors], fmt='o', color='deeppink')
#ax2.errorbar(Rn222Dates, Rn222ELifeValueDeviations,  xerr = [Rn222DateErrorLowers,Rn222DateErrorUppers], yerr=[Rn222ELifeValueDeviationErrors,Rn222ELifeValueDeviationErrors], fmt='o', color='gold')

#ax2.set_xlim([XLimLow, XLimUp])
#ax2.set_ylim([-20, 20])

XLimLow = dt.datetime.fromtimestamp(1489144526)
XLimUp = dt.datetime(2017, 4, 10)

#ax.grid(True)
ax.set_xlim([XLimLow, XLimUp])
ax.set_ylim([500, 650])
#ax.set_ylim([450, 670])
from collections import OrderedDict

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
del by_label['Best-fit trend']
del by_label['$\\pm 1 \\sigma$ C.L. region']
plt.legend(by_label.values(), by_label.keys(), ncol=2, loc = 'lower right',prop={'size':20})
#ax.legend(loc = 'lower right',prop={'size':20})
ax.set_xlabel('Date', fontsize=30)
ax.set_ylabel('Electron lifetime $[\\mu s]$', fontsize=30)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
plt.tight_layout()




ax.set_xlim([XLimLow, XLimUp])

fig.autofmt_xdate()

plt.savefig(FigureSaveName+".png", format='png')
plt.savefig(FigureSaveName+".pdf", format='pdf')

plt.show()
