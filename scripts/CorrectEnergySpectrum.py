import pickle
import numpy as np
import scipy
from scipy.interpolate import interp1d
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import MCMC_Tools
import FormPars

import lax
from lax.lichens import sciencerun0


ScienceRun     = 0
EnergiesToPlot = [41.5, 163.9, 236.2, 583.2, 609.3, 1173.2, 1332.5, 1764.5, 2204.1, 2614.5]
PathToData     = '/project/lgrandi/kobayashi1/work/xenon/analysis/ElectronLifetime/data/SR0_with_lax_cuts.pkl'

w = 0.0137
if ScienceRun == 0:
    g1  = 0.1442
    g2b = 11.52
#elif ScienceRun == 1:
#    g1  = 0.1469
#    g2b = 11.13


PathToLifetimePredictions = '/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/MCMC_Results/TXTs/Prediction_170711.txt'


def GetKrInterpolator(PathToLifetimePredictions, ChangeVal, ChangeValErr, **kwargs):
    (
        PredictionUnixtimes,
        Rn222PredictedELifes,
        Rn222PredictedELifeLows,
        Rn222PredictedELifeUps,
        Rn222PredictedELifeLowErrs,
        Rn222PredictedELifeUpErrs) = MCMC_Tools.LoadPredictions(PathToLifetimePredictions, **kwargs)


    # change to Kr
    (
        KrPredictedELifes,
        KrPredictedELifeLows,
        KrPredictedELifeUps,
        KrPredictedELifeLowErrs,
        KrPredictedELifeUpErrs) = MCMC_Tools.ChangeElectronLifetime(Rn222PredictedELifes,
                                                                    Rn222PredictedELifeLows,
                                                                    Rn222PredictedELifeUps,
                                                                    Rn222PredictedELifeLowErrs,
                                                                    Rn222PredictedELifeUpErrs,
                                                                    ChangeVal,
                                                                    ChangeValErr
                                                                    )

    KrPredictionInterpolator = interp1d(PredictionUnixtimes, KrPredictedELifes)
    return KrPredictionInterpolator







ChangeVal, ChangeValErr = FormPars.GetKrCorrection()

print('\nKr83m interpolater loading')
KrPredictionInterpolator = GetKrInterpolator(PathToLifetimePredictions, ChangeVal, ChangeValErr)

print('\nLoading data from ' + PathToData)
data                  = pickle.load(open(PathToData, 'rb'))
data['dt']            = data.drift_time/1e3


s2_corr_kr            = np.exp(data.dt.values / KrPredictionInterpolator(data.event_time.values/1e9))
data['E']             = (data.cs1/g1 + data.cs2_bottom/g2b) * w
data['cs2_bottom_kr'] = data['s2'] * (1 - data['s2_area_fraction_top']) * s2_corr_kr
data['E_kr']             = (data.cs1 / g1 + data.cs2_bottom_kr / g2b) * w

cuts = [
            sciencerun0.S2AreaFractionTop(),
#            sciencerun0.S1AreaFractionTop(),
            sciencerun0.S2Width(),
            sciencerun0.S1SingleScatter(),
            sciencerun0.S2SingleScatter(),
            sciencerun0.S2Threshold(),
            sciencerun0.DAQVeto(),
#            sciencerun0.PreS2Junk(),
#            sciencerun0.SingleElectronsS2(),
            sciencerun0.InteractionExists(),
            sciencerun0.InteractionPeaksBiggest(),
#            sciencerun0.S2Tails(),
            sciencerun0.S2PatternLikelihood()
            ]

cutnames = ['Cut' + cut.__class__.__name__ for cut in cuts]

print('\nApplying cuts')
for cut in cuts:
    data = cut.process(data)


X = (
        (data.CutS2AreaFractionTop)         &
#        (data.CutS1AreaFractionTop)         &
        (data.CutS2Width)                   &
        (data.CutS1SingleScatter)           &
        (data.CutS2SingleScatter)           &
        (data.CutS2Threshold)               &
        (data.CutDAQVeto)                   &
#        (data.Cut.PreS2Junk)                &
#        (data.Cut.SingleElectronsS2)        &
#        (data.CutS2Tails)                   &
#        (data.CutS2PatternLikelihood)       &
        (data.CutInteractionExists)         &
        (data.CutInteractionPeaksBiggest)   )

data_backup = data
data = data[X]






fig, ax = plt.subplots(2, 1, figsize=(15, 10))
ax, ax2 = ax

ax.hist2d(data.E, data.z, bins=(300, 300), range=((0, 3000), (-90, -10)), cmap='viridis', norm=colors.LogNorm())
for energy in EnergiesToPlot:
    ax.axvline(x=energy, color='r')

ax.set_title('Combined energy spectrum from Rn and Kr lifetime corrections')
ax.set_xlabel('Energy - from Rn correction [keV]')
ax.set_ylabel('z [cm]')

ax2.hist2d(data.E_kr, data.z, bins=(300, 300), range=((0, 3000), (-90, -10)), cmap='viridis', norm=colors.LogNorm())
for energy in EnergiesToPlot:
    ax2.axvline(x=energy, color='r')

ax2.set_xlabel('Energy - from Kr correction [keV]')
ax2.set_ylabel('z [cm]')

#plt.savefig('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/Figures/Corrections/CES/Rn_vs_Kr_energy.png', format='png')
#plt.savefig('/home/kobayashi1/work/xenon/analysis/ElectronLifetime/ElectronLifetime/Figures/Corrections/CES/Rn_vs_Kr_energy.pdf', format='pdf')

plt.show()

