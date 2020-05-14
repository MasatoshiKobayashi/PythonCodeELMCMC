#cython: language_level=3
import ImpurityTrend
from ImpurityTrend import *

#import ROOT
import re
import numpy as np
import scipy as sp
import os, sys
from scipy.interpolate import interp1d

class MyElectronLifetimeTrend:

    def __init__(self, HistorianFile, MinUnixTime, MaxUnixTime, pars):
        # need 1 par3 for electron lifetime trend
        # need 8 pars for impurity trend
        # 2016-12-12 need either 13 or 14 parss
        pars_electronlifetime = pars[0:1]
        pars_impurity = pars[1:]
        self.ImpurityTrend = MyImpurityTrend(HistorianFile, MinUnixTime, MaxUnixTime, pars_impurity)
        self.SetDefaultElectronLifetimeParameters(pars_electronlifetime)
        return

    def SetDefaultElectronLifetimeParameters(self, pars):
        if not len(pars)==1:
            raise ValueError("Parameters for default electron lifetime setting not enough!")
        self.ImpurityAttachingRate = pars[0]
        return

    def SetParameters(self, pars):
        self.ImpurityAttachingRate = np.abs(pars[0])
        if self.ImpurityAttachingRate<1e-40:
            self.ImpurityAttachingRate = 1e-40
        pars_impurity = pars[1:]
        self.ImpurityTrend.SetParameters(pars_impurity)
        return    

    def GetParameters(self):
        pars =  [
                      self.ImpurityAttachingRate
                     ]
        pars.extend(self.ImpurityTrend.GetParameters())
        return pars

    def GetImpurityTrend(self):
        return self.ImpurityTrend

    #def GetElectronLifetime(self, unixtimes):
    def GetElectronLifetime(self, unixtimes,Error,SysAttach):
        IntErr=False
        if not hasattr(unixtimes, '__iter__'):
            unixtimes = np.asarray([unixtimes])
            Error = np.asarray([Error])
        # the electron lifetime trend model
        Ig, Il, LN2 = self.ImpurityTrend.GetConcentrationsAve(unixtimes,Error)
        AttachingRateCorrection = self.ImpurityTrend.GetAttachingRateCorrectionFactor(unixtimes,SysAttach)
        if(Il[0]==Il[-1]):
            IntErr=True
        if 0 in AttachingRateCorrection:
            return np.inf
        if self.ImpurityAttachingRate==0:
            return np.inf
        return 1./self.ImpurityAttachingRate/AttachingRateCorrection/Il,IntErr

    #def GetImpurities(self, unixtimes):
    def GetImpurities(self, unixtimes,Error):
        if not hasattr(unixtimes, '__iter__'):
            unixtimes = np.asarray([unixtimes])
            Error = np.asarray([Error])
        # the electron lifetime trend model
        Ig, Il,LN2 = self.ImpurityTrend.GetConcentrationsAve(unixtimes,Error)
        return Ig, Il,LN2

    def GetOutgassing(self, unixtimes,Error):
        if not hasattr(unixtimes, '__iter__'):
            unixtimes = np.asarray([unixtimes])
            Error = np.asarray([Error])
        # the electron lifetime trend model
        Ig, Il = self.ImpurityTrend.GetOutgassing(unixtimes)
        return Ig, Il


    def GetFlows(self, unixtimes):
        if not hasattr(unixtimes, '__iter__'):
            unixtimes = np.asarray([unixtimes])
        # the electron lifetime trend model
        Fg, Fl= self.ImpurityTrend.GetFlows(unixtimes)
        return Fg, Fl


    def GetPhaseEx(self, unixtimes,Error):
        if not hasattr(unixtimes, '__iter__'):
            unixtimes = np.asarray([unixtimes])
            Error = np.asarray([Error])
        # the electron lifetime trend model
        Vapo, Cond,FlowCond = self.ImpurityTrend.GetPhaseEx(unixtimes)
        return Vapo,Cond,FlowCond


    def ManualChangingFlows(self, NewLiquidFlow, NewGasFlow, start_unixtime, end_unixtime):
        self.ImpurityTrend.ManualChangingFlows(NewLiquidFlow, NewGasFlow, start_unixtime, end_unixtime)
        return
