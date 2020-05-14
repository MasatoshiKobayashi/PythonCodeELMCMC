#cython: language_level=3
import numpy as np

import time
import copy
import click
import emcee
import argparse
import sys

import ElectronLifetimeTrend
from ElectronLifetimeTrend import *

import FormPars
from FormPars import *

import MCMC_Tools
from MCMC_Tools import *

import LnLike

class MyFit_func:

    def __init__(self):
        return

    def Init(self,ConditionSets,FileSets,DataSets):
        self.FormPars = MyFormPars()
        self.RegisteredPenalties = [[],[],[]]
        self.CorrectionTimingSets = []
        self.CorrectionTypes = []
        self.ElectronLifetimeData = {}
        self.SetGlobalPars(*ConditionSets)
        self.SetFiles( *FileSets)
        self.SetELifeDataDiv( *DataSets)
        LnLike.SetFunc(self)
        return


    def GetElectronLifetimeData(self):
        return self.ElectronLifetimeData
    
    def SetFiles(self, historian_file,  output_file,output_txt, Pars, ModeMinuit=False):
        if(self.CheckPars<1):
            raise ValueError("Please set GlobalPars First.")
    
        self.FitOutput     = output_file
        self.FitOutputTXT     = output_txt
        self.HistorianFile = historian_file
    
        # initial parameters
        self.Param_file     = Pars
        if ModeMinuit:
            x0 = MCMC_Tools.LoadParams(Pars)
            dum, x0_steps = self.FormPars.GetOldBestParameters()
        else:
            x0, x0_steps = self.FormPars.GetOldBestParameters()
        initial_pars,IsOutOfBoundary,ListOfIndex=self.FormPars.FormPars(x0)
    
        if IsOutOfBoundary:
            Limits = self.FormPars.GetParLimits()
            print('Out of Boundary for initial parameters! '+str(ListOfIndex))
            for i in range(len(ListOfIndex)):
                index=ListOfIndex[i]
                iLimits=Limits[index]
                ipar=x0[index]
                print('Parameter '+str(ListOfIndex)+": "+str(ipar)+", Limit:"+str(iLimits[0])+", "+str(iLimits[1]))
            raise ValueError('Out of Boundary for initial parameters!'+str(ListOfIndex))
    
        # Initialize electron lifetime trend
        self.pElectronLifetimeTrend = ElectronLifetimeTrend.MyElectronLifetimeTrend(self.HistorianFile, self.MinUnixTime, self.MaxUnixTime, initial_pars)
    
        self.CheckFiles=1
    
        return x0,x0_steps 
    
    def SetGlobalPars(self,int  in_nwalkers, int in_niterations, int in_nthreads, int in_include_po_rn, float in_min_unixtime, float in_max_unixtime, int in_verbose, int in_include_firstss):
    
        self.nwalkers     = in_nwalkers
        self.niterations  = in_niterations
        self.nthreads     = in_nthreads
        self.IncludePoRn  = in_include_po_rn
        self.IncludeFirstSS  = in_include_firstss
        self.MinUnixTime  = in_min_unixtime
        self.MaxUnixTime  = in_max_unixtime
        self.verbose      = in_verbose
    
        self.CheckPars=1
    
        return 
    
    def SetELifeDataDiv(self, inSSELifeDataFile,inPoRnELifeDataFile,inPo218ELifeDataFile,inRn220ELifeDataFile,inRn222ELifeDataFile):
    
        if(self.CheckPars<1):
            raise ValueError("Please set GlobalPars First.")
    
        if(self.CheckFiles<1):
            raise ValueError("Please set HistorianFiles First.")
    
        
        self.SSELifeDataFile     =inSSELifeDataFile    
        self.PoRnELifeDataFile   =inPoRnELifeDataFile  
        self.Po218ELifeDataFile  =inPo218ELifeDataFile 
        self.Rn220ELifeDataFile  =inRn220ELifeDataFile 
        self.Rn222ELifeDataFile  =inRn222ELifeDataFile 
    
        # load electron lifetime data
        (   SSUnixTimes,
            SSUnixTimeErrors,
            SSELifeValues,
            SSELifeValueErrors) = MCMC_Tools.LoadRawFitData('SingleScatter', PathToFile=self.SSELifeDataFile)
        
        (   PoRnUnixTimes,
            PoRnUnixTimeErrors,
            PoRnELifeValues,
            PoRnELifeValueErrors) = MCMC_Tools.LoadRawFitData('PoRn', PathToFile=self.PoRnELifeDataFile)
        
        # currently does not use Po-218 in fit
        (   Po218UnixTimes,
            Po218UnixTimeErrors,
            Po218ELifeValues,
            Po218ELifeValueErrors) = MCMC_Tools.LoadRawFitData('Po218', PathToFile=self.Po218ELifeDataFile)
        
        (   Rn220UnixTimes,
            Rn220UnixTimeErrors,
            Rn220ELifeValues,
            Rn220ELifeValueErrors) = MCMC_Tools.LoadRawFitData('Rn220', PathToFile=self.Rn220ELifeDataFile)
        
        (   Rn222UnixTimes,
            Rn222UnixTimeErrors,
            Rn222ELifeValues,
            Rn222ELifeValueErrors) = MCMC_Tools.LoadRawFitData('Rn222', PathToFile=self.Rn222ELifeDataFile)
        
        SR1UnixTime = 1485907200
        WTUnixTime = 1468886400
        OldPAXUnixTime = self.FormPars.GetOldPAXUnixTime() # Use this to sepalate the data based on the voltage
        VoltageChangeUnixTime = self.FormPars.GetVoltageChangeUnixTime() # Use this to sepalate the data based on the voltage
        PoRnMinUnixTime = np.min(PoRnUnixTimes)
        Rn222MinUnixTime = np.min(Rn222UnixTimes)


        #Appry correction for SinleScatterDatas
        self.CorrectionTimingSets.append([self.MinUnixTime,WTUnixTime])
        self.CorrectionTypes.append('SS')

        #Appry correction for SinleScatterDatas
        self.CorrectionTimingSets.append([WTUnixTime,PoRnMinUnixTime])
        self.CorrectionTypes.append('SS')

        #Appry correction for old PAX
        self.CorrectionTimingSets.append([PoRnMinUnixTime,OldPAXUnixTime])
        self.CorrectionTypes.append('PAX')
 
        #Appry correction for PoRn data
        self.CorrectionTimingSets.append([PoRnMinUnixTime,Rn222MinUnixTime])
        self.CorrectionTypes.append('PoRn')

        #Appry correction for before 15kV data
        self.CorrectionTimingSets.append([self.MinUnixTime,VoltageChangeUnixTime])
        self.CorrectionTypes.append('Field150')

        #Appry correction for SR1Data + SR2Data (Consider correction are same for SR1 amd SR2)
        self.CorrectionTimingSets.append([VoltageChangeUnixTime,SR1UnixTime])
        self.CorrectionTypes.append('FieldSR0')

        #Appry correction for SR1Data + SR2Data (Consider correction are same for SR1 amd SR2)
        self.CorrectionTimingSets.append([SR1UnixTime,self.MaxUnixTime])
        self.CorrectionTypes.append('FieldSR1')

        AllUnixTimes = copy.deepcopy(SSUnixTimes)
        AllUnixTimeErrors = copy.deepcopy(SSUnixTimeErrors)
        AllELifeValues = copy.deepcopy(SSELifeValues)
        AllELifeValueErrors = copy.deepcopy(SSELifeValueErrors)
       
        # Remove values that has unixtime larger than the first one in PoRnUnixTimes
        if self.IncludePoRn:
            indicesToKeep = np.where(AllUnixTimes < PoRnMinUnixTime )[0]
            AllUnixTimes = np.append(AllUnixTimes[indicesToKeep], PoRnUnixTimes)
            AllUnixTimeErrors = np.append(AllUnixTimeErrors[indicesToKeep],PoRnUnixTimeErrors)
            AllELifeValues = np.append(AllELifeValues[indicesToKeep],PoRnELifeValues)
            AllELifeValueErrors = np.append(AllELifeValueErrors[indicesToKeep],PoRnELifeValueErrors)

        # And then extend the list with Rn222
        indicesToKeep = np.where(AllUnixTimes < Rn222MinUnixTime )[0]
        AllUnixTimes = np.append(AllUnixTimes[indicesToKeep],Rn222UnixTimes)
        AllUnixTimeErrors = np.append(AllUnixTimeErrors[indicesToKeep],Rn222UnixTimeErrors)
        AllELifeValues = np.append(AllELifeValues[indicesToKeep],Rn222ELifeValues)
        AllELifeValueErrors = np.append(AllELifeValueErrors[indicesToKeep],Rn222ELifeValueErrors)
    
        # Rn-220 calibration during power drop in February 2018
        # not background data during this period
        AllUnixTimes = np.append(AllUnixTimes,Rn220UnixTimes)
        AllUnixTimeErrors = np.append(AllUnixTimeErrors,Rn220UnixTimeErrors)
        AllELifeValues = np.append(AllELifeValues,Rn220ELifeValues)
        AllELifeValueErrors =np.append(AllELifeValueErrors,Rn220ELifeValueErrors)

        # sort values according to unixtime
        AllELifeValues = np.asarray([x for _,x in sorted(zip(AllUnixTimes, AllELifeValues), key=lambda pair: pair[0])])
        AllELifeValueErrors =np.asarray( [x for _,x in sorted(zip(AllUnixTimes, AllELifeValueErrors), key=lambda pair: pair[0])])
        AllUnixTimeErrors = np.asarray([x for _,x in sorted(zip(AllUnixTimes, AllUnixTimeErrors), key=lambda pair: pair[0])])
        AllUnixTimes = np.asarray(sorted(AllUnixTimes.tolist()))
   
        if not self.IncludeFirstSS:
            # currently do not fit small region before magnetic pump upgrade
            indicesToKeep = np.where((AllUnixTimes > 1467417600))[0]
            AllUnixTimes = np.asarray(AllUnixTimes)[indicesToKeep]
            AllUnixTimeErrors = np.asarray(AllUnixTimeErrors)[indicesToKeep]
            AllELifeValues = np.asarray(AllELifeValues)[indicesToKeep]
            AllELifeValueErrors = np.asarray(AllELifeValueErrors)[indicesToKeep]
 
        # Fit Only between Min/Max UnixTime 
        indicesToKeep = np.where((AllUnixTimes < self.MaxUnixTime) & (AllUnixTimes > self.MinUnixTime))[0]
        AllUnixTimes = np.asarray(AllUnixTimes)[indicesToKeep]
        AllUnixTimeErrors = np.asarray(AllUnixTimeErrors)[indicesToKeep]
        AllELifeValues = np.asarray(AllELifeValues)[indicesToKeep]
        AllELifeValueErrors = np.asarray(AllELifeValueErrors)[indicesToKeep]
    
        self.ElectronLifetimeData['UnixTimes'] = AllUnixTimes
        self.ElectronLifetimeData['UnixTimeErrors'] = AllUnixTimeErrors
        self.ElectronLifetimeData['Values'] = AllELifeValues
        self.ElectronLifetimeData['ValueErrors'] = AllELifeValueErrors
    
    
        #Registering the Penalty terms for the Chi2 calculation
        self.RegisterPenalty(-1,0,1) #Field SR1
        self.RegisterPenalty(-2,0,1) #Field SR0
        self.RegisterPenalty(-3,0,1) #Field SRB
        self.RegisterPenalty(-4,0,1) #PoRn
        self.RegisterPenalty(-5,0,1) #PAX
        #self.RegisterPenalty(-6,0,1) #SS
        self.RegisterPenalty(-8,258,234) #Volatility
        self.RegisterPenalty(1,0.285,0.053) #Phi
        self.RegisterPenalty(3,0,0.11) #Lambda_g/Lambda_L ratio


        self.CheckData=1

    
        return self.ElectronLifetimeData
    
    def GetFlows(self,unixtimes):
        Fg,Fl = self.pElectronLifetimeTrend.GetFlows(unixtimes)
    
        return Fg,Fl
 
    def GetExpectedLifetimes(self,pars):
        UnixTimes = self.ElectronLifetimeData['UnixTimes']
        UnixTimeErrors = self.ElectronLifetimeData['UnixTimeErrors']
        Values = self.ElectronLifetimeData['Values']
        ValueErrors = self.ElectronLifetimeData['ValueErrors']
        self.ElectronLifetimeData['UnixTimesChi2'] = UnixTimes
    
        self.pElectronLifetimeTrend.SetParameters(pars)
        alphas = pars[-1]
        alphaAttach = 0.0
        expected,IntErr = self.pElectronLifetimeTrend.GetElectronLifetime(UnixTimes,UnixTimeErrors,alphaAttach)
    
        for i,(CorrectionType, CorrectionTiming) in enumerate(zip(self.CorrectionTypes,self.CorrectionTimingSets)):
            Indices = np.where((CorrectionTiming[0]<=UnixTimes) & (UnixTimes<CorrectionTiming[1]))[0]
            expected = MCMC_Tools.ApplyFuncCorrection(expected,Indices,alphas[i],CorrectionType,UnixTimes)
    
        return expected, Values, ValueErrors,IntErr

    def Randomize(self,UnixTimes,Pars_Trials,TrueLifeTime=False):
        # Initial the 1 sigma lower/upper of the trend
        N = len(UnixTimes)
        Trends =[ [] for i in range(N) ] 
        ImpurityGXe =[ [] for i in range(N) ] 
        ImpurityLXe =[ [] for i in range(N) ] 
        ImpurityLN2 =[ [] for i in range(N) ] 
        OutgassingGXe =[ [] for i in range(N) ] 
        OutgassingLXe =[ [] for i in range(N) ] 
        VapoTerm = [ [] for i in range(N) ] 
        CondTerm =[ [] for i in range(N) ] 
        FlowCondTerm =[ [] for i in range(N) ] 

        Nlost=0
    
        Trends_chi2 = []
        Trends_Rchi2 = []
        Trends_chi2_tot = []
        Trends_Rchi2_tot = []
        pars, IfSth, parindex = self.FormPars.FormPars(Pars_Trials[0])
        pre_chi2 = self.SetLnLikeDataCorrections(pars)
        UnixTimes_chi2 = self.ElectronLifetimeData['UnixTimesChi2']
        for i in range(len(UnixTimes_chi2)):
            Trends_chi2.append([])
            Trends_Rchi2.append([])
        Nparam = len(Pars_Trials[0])
        NDF = len(UnixTimes_chi2) - Nparam
    
        # pickup randomly the pars in the MCMC walker
        for i, pars_random in enumerate(Pars_Trials):
            if(i%50==0):
                print("i = "+str(i))
            pars, IfSth, parindex = self.FormPars.FormPars(pars_random)
            pre_trends,Ig,Il,LN2,Og,Ol,Vapo,Cond,FlowCond=self.GetTrends(UnixTimes,pars_random,TrueLifeTime)
            pre_chi2 = self.SetLnLikeDataCorrections(pars)
            Penalty = self.CalcPenalty(pars_random)
            chi2_tot = np.sum(pre_chi2) + Penalty
            if (np.any(np.asarray(pre_trends)<0)):
                Nlost+=1
            else: 
                for j in range(0,len(UnixTimes)):
                    Trends[j].append(pre_trends[j])
                    ImpurityGXe[j].append(Ig[j])
                    ImpurityLXe[j].append(Il[j])
                    ImpurityLN2[j].append(LN2[j])
                    OutgassingGXe[j].append(Og[j])
                    OutgassingLXe[j].append(Ol[j])
                    VapoTerm[j].append(Vapo[j])
                    CondTerm[j].append(Cond[j])
                    FlowCondTerm[j].append(FlowCond[j])
                chi2_cum = 0    
                Npoints = 0.0
                for j in range(0,len(UnixTimes_chi2)):
                    Trends_chi2[j].append(pre_chi2[j])
                    chi2_cum = chi2_cum+pre_chi2[j]
                    Npoints = Npoints+1.0
                    if(Npoints>Nparam):
                        RChi2 = chi2_cum/(Npoints-Nparam)
                    else:
                        RChi2 = 0
                    Trends_Rchi2[j].append(RChi2)
                Trends_chi2_tot.append(chi2_tot)
                Trends_Rchi2_tot.append(chi2_tot/NDF)
    
        return Trends,ImpurityGXe,ImpurityLXe,ImpurityLN2,OutgassingGXe,OutgassingLXe,VapoTerm,CondTerm,FlowCondTerm, Nlost, UnixTimes_chi2,Trends_chi2, Trends_chi2_tot,Trends_Rchi2, Trends_Rchi2_tot
    
    def GetTrends(self,UnixTimes,Pars,TrueLifeTime=False):
        # Initial the 1 sigma lower/upper of the trend
        Trends = []
        ImpurityGXe = []
        ImpurityLXe = []
        ImpurityLN2 = []
        OutgassingGXe = []
        OutgassingLXe = []
        VapoTerm = []
        CondTerm = []
        FlowCondTerm = []
        
        # pickup randomly the pars in the MCMC walker
        pars, IfSth, parindex = self.FormPars.FormPars(Pars)
        self.pElectronLifetimeTrend.SetParameters(pars)
        alphas = pars[-1]
        alphaAttach=0.0
        expected,IntErr = self.pElectronLifetimeTrend.GetElectronLifetime(UnixTimes,np.ones(len(UnixTimes))*86400,alphaAttach) #Currenly xerror is not used
        Ig,Il,LN2=self.pElectronLifetimeTrend.GetImpurities(UnixTimes,np.ones(len(UnixTimes))*86400*2)
        Lambda_g,Lambda_l=self.pElectronLifetimeTrend.GetOutgassing(UnixTimes,np.ones(len(UnixTimes))*86400*2)
        Vapo,Cond,FlowCond=self.pElectronLifetimeTrend.GetPhaseEx(UnixTimes,np.ones(len(UnixTimes))*86400*2)
        if not (TrueLifeTime):
            for i,(CorrectionType, CorrectionTiming) in enumerate(zip(self.CorrectionTypes,self.CorrectionTimingSets)):
                Indices = np.where((CorrectionTiming[0]<=UnixTimes) & (UnixTimes<CorrectionTiming[1]))[0]
                expected = MCMC_Tools.ApplyFuncCorrection(expected,Indices,alphas[i],CorrectionType,UnixTimes)
        for j in range(0,len(UnixTimes)):
            Trends.append(expected[j])
            ImpurityGXe.append(Ig[j])
            ImpurityLXe.append(Il[j])
            ImpurityLN2.append(LN2[j])
            OutgassingGXe.append(Lambda_g[j])
            OutgassingLXe.append(Lambda_l[j])
            VapoTerm.append(Vapo[j])
            CondTerm.append(Cond[j])
            FlowCondTerm.append(FlowCond[j])
        return Trends,ImpurityGXe,ImpurityLXe,ImpurityLN2,OutgassingGXe,OutgassingLXe,VapoTerm, CondTerm,FlowCondTerm

    
    def SetLnLikeDataCorrections(self,pars):
        expected, Values, ValueErrors,IntErr = self.GetExpectedLifetimes(pars)
        self.LnL_each = np.power((Values - expected)/ValueErrors, 2.)
    
        return self.LnL_each 

    def LnLikeDataCorrections(self,x):
        pars, IfOutOfBoundary,parindex = FormPars.FormPars(x)
        expected, Values, ValueErrors,IntErr = self.GetExpectedLifetimes(pars)
        if IfOutOfBoundary or np.any(expected!=expected):
            if IntErr:
                print("Integration error!")
            return -np.inf
        if IntErr:
            print("Integration error!")
            return -np.inf
        cdef float LnL = 0.
        LnL_arr = 0.
        LnL_arr += -0.5* np.power((Values - expected)/ValueErrors, 2.)
        LnL = LnL_arr.sum()
        Penalty = self.CalcPenalty(x)
        LnL-=Penalty/2.
    
        return LnL

    def RegisterPenalty(self,index,mu,sigma):
        self.RegisteredPenalties[0].append(index)
        self.RegisteredPenalties[1].append(mu)
        self.RegisteredPenalties[2].append(sigma)

        return

    def CalcPenalty(self,pars):
        RegisteredIndex = np.asarray(self.RegisteredPenalties[0])
        RegisteredPars = np.asarray(pars)[RegisteredIndex]
        RegisteredMu = np.asarray(self.RegisteredPenalties[1])
        RegisteredSigma = np.asarray(self.RegisteredPenalties[2])

        return np.sum(((RegisteredPars - RegisteredMu)/RegisteredSigma)**2)
        
