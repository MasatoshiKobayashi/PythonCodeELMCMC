#cython: language_level=3
import HistorianData
from HistorianData import *

#import ROOT
import re
import numpy as np
import scipy as sp
import os, sys
from scipy.interpolate import interp1d
import time
from datetime import datetime
import FormPars
from FormPars import *

class MyImpurityTrend:

    def __init__(self, HistorianFile, MinUnixTime, MaxUnixTime, pars):
        Fields = [0.0583245443043, 0.0787113537639, 0.117382423495, 0.162989973122, 0.253660532825, 0.34227580980800004, 0.436223335797, 0.588573683763, 0.7941034911479999, 1.0868133019500001, 1.38497072037, 1.81605663549, 2.41539631008, 3.44978922077, 3.9223289353700004, 5.8466333187299995, 7.24083005596]
        AttachingRates = [173839448697.0, 166490987263.0, 154902864208.0, 144105038741.0, 124672767516.0, 112649545428.0, 103268822098.0, 90632673738.3, 78393350018.2, 67808393527.7, 59505871959.5, 52222275531.1, 44516323097.2, 35805292980.3, 33787097454.0, 26016257301.4, 22829772653.1]
        AttachingRateErrors = [0.08742831e11, 0.08633761e11, 0.08431171e11, 0.08199191e11, 0.07759951e11, 0.07357911e11, 0.06960131e11, 0.06374441e11, 0.05695061e11, 0.0493191e11, 0.04380841e11, 0.03932181e11, 0.03804721e11, 0.04174731e11, 0.04394691e11, 0.05064351e11, 0.05294711e11]
        self.FormPars = MyFormPars()
        self.ReferenceField = 0.15 
        self.AttachingRateAsField = interp1d(Fields, AttachingRates, fill_value = 'extrapolate')
        self.AttachingRateErrorAsField = interp1d(Fields, AttachingRateErrors, fill_value = 'extrapolate')
        self.HistorianData = MyHistorianData(HistorianFile)
        self.SetNuisanceParameters()
        self.SetTimeWindow(MinUnixTime, MaxUnixTime)
        self.SetDefaultParameters()
        self.GetGlobalHistorianData()
        self.GetGlobalDrops()
        self.SetParameters(pars)
        return

    def __eq__(self, other):
        # SC handler
        self.HistorianData = other.HistorianData
        self.GlobalDays = other.GlobalDays
        self.GlobalLiquidFlows = other.GlobalLiquidFlows
        self.GlobalGasFlows = other.GlobalGasFlows
        self.GlobalCoolingPowers = other.GlobalCoolingPowers
        self.GlobalCathodeVoltages = other.GlobalCathodeVoltages
        # time window
        self.MinUnixTime = other.MinUnixTime
        self.MaxUnixTime = other.MaxUnixTime
        # nuisance
        self.ReferenceField = other.ReferenceField
        self.MassGXe = other.MassGXe
        self.MassLXe = other.MassLXe
        self.LatentHeatXenon = other.LatentHeatXenon
        self.StandardXenonDensity = other.StandardXenonDensity
        self.StableCoolingPower = other.StableCoolingPower
        self.DefaultTimeStep = other.DefaultTimeStep
        self.DefaultTimeStepGasOnly = other.DefaultTimeStepGasOnly
        # parameters
        self.InitialConcentrationGXe = other.InitialConcentrationGXe
        self.InitialConcentrationLXe = other.InitialConcentrationLXe
        self.ImpurityAttachingProbVaporization = other.ImpurityAttachingProbVaporization
        self.ImpurityAttachingProbCondensation = other.ImpurityAttachingProbCondensation
        self.OutgassingRateGXe = other.OutgassingRateGXe
        self.OutgassingRateLXe = other.OutgassingRateLXe
        self.ImpurityChangingUnixTimes = other.ImpurityChangingUnixTimes
        self.ImpurityConcentrationChangesGXe = other.ImpurityConcentrationChangesGXe
        self.ImpurityConcentrationChangesLXe = other.ImpurityConcentrationChangesLXe
        self.OutgassingRateGXeChanges = other.OutgassingRateGXeChanges # [start/end changing unixtime, changing amount], assuming linear
        self.OutgassingRateGXeFractions = other.OutgassingRateGXeFractions
        self.OutgassingRateLXeFractions = other.OutgassingRateLXeFractions
        # [ start unixtime, linear coefficient]
        # interp1d
        self.inter_ConcentrationsGXe = other.inter_ConcentrationsGXe
        self.inter_ConcentrationsLXe = other.inter_ConcentrationsLXe
        self.AttachingRateAsField = other.AttachingRateAsField
        return

    def SetTimeWindow(self, MinUnixTime, MaxUnixTime):
        self.MinUnixTime = MinUnixTime
        self.MaxUnixTime = MaxUnixTime
        return

    def GetGlobalHistorianData(self):
        cdef int Npoints1, Npoints2, Npoints3
        cdef float unixtime 
        GasOnlyPeriod = self.FormPars.GetGasOnlyPeriods()
        GasOnlyStartUnixtime = GasOnlyPeriod[0][0]
        GasOnlyEndUnixtime = GasOnlyPeriod[0][1]

        Npoints1 = (GasOnlyStartUnixtime - self.MinUnixTime) / 3600. / 24. / self.DefaultTimeStep + 1
        if (Npoints1%2==0): #Need to be odd number
            Npoints1+=1
        Npoints2 = (GasOnlyEndUnixtime - GasOnlyStartUnixtime ) / 3600. / 24. / self.DefaultTimeStepGasOnly 
        if (Npoints2%2==0): #odd number
            Npoints2+=1

        Npoints3 = (self.MaxUnixTime - GasOnlyEndUnixtime ) / 3600. / 24. / self.DefaultTimeStep 
        if (Npoints3%2==0): #odd number
            Npoints3+=1
        self.Npoints=Npoints1+Npoints2+Npoints3-2

        GlobalDays1 = np.linspace(
                                         (self.MinUnixTime -self.MinUnixTime   )/3600./24.,
                                         (GasOnlyStartUnixtime - self.MinUnixTime )/3600./24.,
                                         int(Npoints1)
                                        )

        GlobalDays2 = np.linspace(
                                         (GasOnlyStartUnixtime  - self.MinUnixTime )/3600./24.,
                                         (GasOnlyEndUnixtime  - self.MinUnixTime  )/3600./24.,
                                         int(Npoints2)
                                        )
        GlobalDays3 = np.linspace(
                                         (GasOnlyEndUnixtime  - self.MinUnixTime  )/3600./24.,
                                         ( self.MaxUnixTime  - self.MinUnixTime )/3600./24.,
                                         int(Npoints3)
                                        )
        self.GlobalDays = np.append(np.append(GlobalDays1,GlobalDays2[1:]),GlobalDays3[1:])
        #print(self.MaxUnixTime, self.MinUnixTime, self.DefaultTimeStep, Npoints)

        GlobalUnixTime1 = np.linspace(
                                         self.MinUnixTime,
                                         GasOnlyStartUnixtime,
                                         int(Npoints1)
                                        )

        GlobalUnixTime2 = np.linspace(
                                         GasOnlyStartUnixtime ,
                                         GasOnlyEndUnixtime,
                                         int(Npoints2)
                                        )
        GlobalUnixTime3 = np.linspace(
                                         GasOnlyEndUnixtime,
                                         self.MaxUnixTime,
                                         int(Npoints3)
                                        )
        self.GlobalUnixTimes = np.append(np.append(GlobalUnixTime1,GlobalUnixTime2[1:]),GlobalUnixTime3[1:])

        self.GlobalLiquidFlows = np.zeros(self.Npoints)
        self.GlobalGasFlows = np.zeros(self.Npoints)
        self.GlobalBellFlows = np.zeros(self.Npoints)
        self.GlobalCoolingPowers = np.zeros(self.Npoints)
        self.GlobalCathodeVoltages = np.zeros(self.Npoints)
        self.GlobalBelowBellTemps = np.zeros(self.Npoints)
        self.GlobalInnerVesselTemps = np.zeros(self.Npoints)
        self.GlobalDropLXe = np.zeros(self.Npoints)
        self.GlobalDropGXe = np.zeros(self.Npoints)
        self.GlobalGXecirc = np.zeros(self.Npoints)
        self.GlobalFlowCondense = np.zeros(self.Npoints)
        self.GlobalHEeff = np.zeros(self.Npoints)

        print(Npoints1,Npoints2,Npoints3)
        # GXe circulation period
        self.GlobalGXecirc[Npoints1:Npoints1+Npoints2]=1

        for i,day in enumerate(self.GlobalDays):
            unixtime = day*3600.*24. + self.MinUnixTime #self.HistorianData.GetReferenceUnixTime()
            LiquidFlow, GasFlow, BellFlow,CoolingPower, CathodeVoltage, BelowBellTemp, InnerVesselTemp = self.HistorianData.GetHistorian(unixtime)
            self.GlobalLiquidFlows[i] = LiquidFlow
            self.GlobalGasFlows[i] = GasFlow
            self.GlobalBellFlows[i] = BellFlow
            self.GlobalCoolingPowers[i] =CoolingPower
            self.GlobalCathodeVoltages[i] = CathodeVoltage
            self.GlobalBelowBellTemps[i] = BelowBellTemp
            self.GlobalInnerVesselTemps[i] = InnerVesselTemp
            if(self.GlobalGXecirc[i]==0 and LiquidFlow>0): # Consider no extra condensation during GXe only curculation term
                self.GlobalFlowCondense[i]=1.0

        self.GlobalTotFlows=self.GlobalLiquidFlows+self.GlobalGasFlows
        p0,p1 = self.FormPars.GetParametersForHE()
        self.GlobalHEeff = np.fmin(100,(p0 + p1*(self.GlobalTotFlows-self.GlobalBellFlows)))/100. * self.GlobalFlowCondense

        return
            
    def UpdateGlobalHistorianData(self):
        # to save time
        #Update the period when there assumes to be a getter deficiency
        Configs = self.HistorianData.GetGetterDeficiencyConfigs()
        self.GlobalBackFractions = np.zeros(self.Npoints)
        for i,Config in enumerate(Configs):
            StartUnixtime = Config[0]
            EndUnixtime = Config[1]
            EndUnixtime_ref=np.fmin(EndUnixtime,self.MaxUnixTime)
            Indices = np.where((StartUnixtime<=self.GlobalUnixTimes) & (self.GlobalUnixTimes<EndUnixtime_ref))[0]
            self.GlobalBackFractions[Indices] = (1-Config[2]) 
        Indices = np.where(self.GlobalLiquidFlows<=0)
        self.GlobalBackFractions[Indices] = 0.0

        return

    def GetGlobalDrops(self):
        # to save time
        # only update the period when there assumes to be a getter deficiency
        # check if need the change for final concentration
        for itime, unixtime_change in enumerate(self.ImpurityChangingUnixTimes):
            StartUnixtime = unixtime_change
            Indices = np.where((StartUnixtime<=self.GlobalUnixTimes))[0]
            StartIndex=Indices[0]
            EndIndex = StartIndex+2
            ChangeIndex = self.ImpurityConcentrationChangeTypes[itime]
            if(ChangeIndex==0):
                self.GlobalDropGXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesGXe[itime])
                self.GlobalDropLXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesLXe[itime])
            elif(ChangeIndex==1):
                self.GlobalDropGXe[StartIndex:EndIndex]=0
                self.GlobalDropLXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesLXe[itime])
            elif(ChangeIndex==2):
                self.GlobalDropGXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesLXe[itime]) * self.RelativeVolatility * (1-self.GlobalHEeff[StartIndex:EndIndex])/ self.GlobalHEeff[StartIndex:EndIndex]
                self.GlobalDropLXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesLXe[itime])
            elif(ChangeIndex==3):
                self.GlobalDropGXe[StartIndex:EndIndex]=np.fabs(self.ImpurityConcentrationChangesGXe[itime])
                self.GlobalDropLXe[StartIndex:EndIndex]=0
        return

    def SetNuisanceParameters(self):
        # totally 4 nuisance parameters are needed
        # the total mass in the three relevant region
        # and the latent heat of Xenon
        # plus the necessary stand xenon density
        # plus the default integration step (1hr)
        self.MassGXe = 23. # kg
        self.MassLXe = 401.+(2756.- 0.) # kg. The LXe mass total
        self.LatentHeatXenon = 95.587e3 # J/kg
        self.StandardXenonDensity = 5.894 # g/L
        self.StableCoolingPower = 140. #W
        self.DefaultTimeStep = 1./30 # days 
        self.DefaultTimeStepGasOnly = 1./168 # days 
        self.OutgassingRateScalingTemperature=(50.6)/(33)#From calculation
        self.OutgassingRateScalingMass=(28.5)/(55.5)#From CAD, PTFE ratio
        self.Time_LXeFilling_Finished =1461801600 #2016/4/28
        self.K = 8.617e-5
        self.Eint = 0.1476 #eV
        self.NInt=2
        return

    def SetDefaultParameters(self):
        self.InitialConcentrationGXe = 100. # ppb
        self.InitialConcentrationLXe = 100. # ppb
        self.RelativeVolatility = 26.0
        self.RelativeCondense = 1. # 100%
        self.CondensationProb = 1. # 100%
        self.OutgassingRateGXe = [1., 0.,1./300,300] # kg*ppb/day
        self.OutgassingRateLXe = [1., 0.,100,1./1000,200] # kg*ppb/day
        self.ImpurityChangingUnixTimes = [] #
        self.ImpurityConcentrationChangesGXe = [] # mol
        self.ImpurityConcentrationChangesLXe = [] # mol
        self.OutgassingRateGXeChanges = []
        self.OutgassingRateGXeFractions = []
        self.OutgassingRateLXeFractions = []
        self.HEeff_all = 1.0
        self.HEeff_lowflow = 1.0
        self.LN2Step = 0.5
        return

    def SetParameters(self, pars):
        # there're 12 parameters needed
        # also will start the calculation if the parameters have been changed
        # @2016-12-12 either 12 or 13 pars
        # @2017-02-07 either 13 or 14 pars
        # @2017-02-13 either 14 or 15 pars
        IfSameAsPrevious = self.CheckIfSame(pars)
        self.InitialConcentrationLXe = pars[0]
        self.RelativeCondense = pars[1]
        self.CondensationProb = pars[2]
        self.OutgassingRateGXe = pars[3]
        self.OutgassingRateLXe = pars[4]
        self.ImpurityChangingUnixTimes = pars[5]
        self.ImpurityConcentrationChangesGXe = pars[6]
        self.ImpurityConcentrationChangesLXe = pars[7]
        self.ImpurityConcentrationChangeTypes = pars[8]
        self.HistorianData.PopOneGetterDeficiencyConfig()
        self.HistorianData.AddOneGetterDeficiencyConfig(pars[9])
        self.RelativeCircGas=    pars[10][0]  
        self.RelativeCircLiquid =pars[10][1] 
        self.RelativeVolatility = pars[11]
        self.VaporizationProb = self.CondensationProb*self.RelativeVolatility
        self.InitialConcentrationGXe = self.InitialConcentrationLXe* self.RelativeVolatility
        self.UpdateGlobalHistorianData()
        self.GetGlobalDrops()
        self.GXeOutgassingRate=self.GetGXeOutgassingRate()
        self.LXeOutgassingRate=self.GetLXeOutgassingRate()
        if not IfSameAsPrevious:
            self.CalculateImpurityConcentration()

        return

    def GetParameters(self):
        return [
                                    self.InitialConcentrationGXe,
                                    self.InitialConcentrationLXe,
                                    self.RelativeVolatility,
                                    self.CondensationProb, 
                                    self.OutgassingRateGXe,
                                    self.OutgassingRateLXe,
                                    self.ImpurityChangingUnixTimes,
                                    self.ImpurityConcentrationChangesGXe,
                                    self.ImpurityConcentrationChangesLXe,
                                    self.OutgassingRateGXeChanges,
                                    self.OutgassingRateGXeFractions,
                                    self.OutgassingRateLXeFractions,
                                    self.HistorianData.GetGetterDeficiencyConfigs(),
                                   ]

    def CheckIfSame(self, pars):
        PreviousPars = self.GetParameters()
        for previous_par, par in zip(PreviousPars, pars):
            if not previous_par==par:
                return False
        return

    def GetHistorianData(self):
        return self.HistorianData

    def GetGXeOutgassingRate(self):
        day=self.GlobalDays
        InnerVesselTemp=self.GlobalInnerVesselTemps
        InnerVesselTempMean = InnerVesselTemp[0]
        GasFlow= self.GlobalGasFlows
        BellFlow=self.GlobalBellFlows
        LInitial1,LExponential1,LInitial2,LExponential2= self.OutgassingRateLXe[0], self.OutgassingRateLXe[1], self.OutgassingRateLXe[2], self.OutgassingRateLXe[3]

        TDiff = (self.MinUnixTime - self.Time_LXeFilling_Finished)/24./3600.
        ErrorScaling= self.OutgassingRateGXe[0]
        TScaleError = self.OutgassingRateScalingTemperature*np.fmax(0,(1+ErrorScaling))
        TimeConstantGas1 =  LExponential1*TScaleError
        TimeConstantGas2 =  LExponential2*TScaleError
        InitialValueGas1 =  LInitial1*self.OutgassingRateScalingMass * TScaleError * np.exp( - TDiff*LExponential1 * (TScaleError - 1) ) #Lambda_g/Lambda_l, at MinUnixTime
        InitialValueGas2 =  LInitial2*self.OutgassingRateScalingMass * TScaleError * np.exp( - TDiff*LExponential2 * (TScaleError - 1) ) #Lambda_g/Lambda_l, at MinUnixTime
        Rate1 = InitialValueGas1*np.exp(-day*TimeConstantGas1)
        Rate2 = InitialValueGas2*np.exp(-day*TimeConstantGas2)
        Rate  = Rate1 + Rate2
        Rate *= np.exp(-self.Eint/(self.K*InnerVesselTemp))/np.exp(-self.Eint/(self.K*InnerVesselTempMean))

        return Rate

    def GetLXeOutgassingRate(self):
        day=self.GlobalDays
        BelowBellTemp=self.GlobalBelowBellTemps
        BellTempMean = BelowBellTemp[0]
        LiquidFlow=self.GlobalLiquidFlows
        Initial1,Exponential1,Initial2,Exponential2= self.OutgassingRateLXe[0], self.OutgassingRateLXe[1], self.OutgassingRateLXe[2], self.OutgassingRateLXe[3]
        Rate = np.exp(-day*Exponential1)*(Initial1) + np.exp(-day*Exponential2)*(Initial2)
        Rate *= np.exp(-self.Eint/(self.K*BelowBellTemp))/np.exp(-self.Eint/(self.K*BellTempMean))

        return Rate

    def GetChangeOfImpurityConcentration(self,int GlobalTimeIndex,PreviousConcentrations):
        cdef:
            #float day, unixtime
            float LiquidFlow,GasFlow,CoolingPower
            float LiquidFlowTerm, GasFlowTerm, VaporizationTerm, CondensationTerm#, GasByPassFlowTerm,LiquidByPassFlowTerm 
            float ConcentrationChangeGXe=0
            float ConcentrationChangeLXe=0
            float PreviousConcentrationGXe=np.fmax(0,PreviousConcentrations[0])
            float PreviousConcentrationLXe=np.fmax(0,PreviousConcentrations[1])
            #float PreviousConcentrationLN2=np.fmax(0,PreviousConcentrations[2])
            ConcentrationChanges = np.zeros(self.NInt,dtype=float)

        # main function for calculating the impurity trend
        day=self.GlobalDays[GlobalTimeIndex]
        unixtime = day*3600.*24. + self.MinUnixTime#self.HistorianData.GetReferenceUnixTime()

        #Getting Global Time-dependent parameters
        LiquidFlow = self.GlobalLiquidFlows[GlobalTimeIndex]
        GasFlow= self.GlobalGasFlows[GlobalTimeIndex]
        BellFlow= self.GlobalBellFlows[GlobalTimeIndex]
        CoolingPower= self.GlobalCoolingPowers[GlobalTimeIndex]
        BackFraction = self.GlobalBackFractions[GlobalTimeIndex]
        InnerVesselTemp=self.GlobalInnerVesselTemps[GlobalTimeIndex]
        BelowBellTemp=self.GlobalBelowBellTemps[GlobalTimeIndex]
        GXecirc = self.GlobalGXecirc[GlobalTimeIndex]
        HEeff = self.GlobalHEeff[GlobalTimeIndex]
        IsFlowCondense = self.GlobalFlowCondense[GlobalTimeIndex]
        RelativeFactor = self.RelativeCondense/(self.RelativeCondense + self.RelativeVolatility*self.CondensationProb*(1 - self.RelativeCondense)) #consider euqiliblium at the tube from cryo to the TPC


        # define GXe and LXe outgassing rate
        ConcentrationChangeGXe = self.GXeOutgassingRate[GlobalTimeIndex]
        ConcentrationChangeLXe = self.LXeOutgassingRate[GlobalTimeIndex]

        ###################
        # the individual terms
        # all in positive 
        ##################
        # Gas flow term
        GasFlowTerm = self.RelativeCircGas*GasFlow*self.StandardXenonDensity*PreviousConcentrationGXe
        GasFlowTerm *= 1e-3*60.*24. # to kg*ppb / day

        # Liquid flow term
        LiquidFlowTerm = self.RelativeCircLiquid*LiquidFlow*self.StandardXenonDensity*PreviousConcentrationLXe
        LiquidFlowTerm *= 1e-3*60.*24.

        # Vaporization
        VaporizationTerm = self.RelativeCondense*(self.StableCoolingPower/self.LatentHeatXenon)*(PreviousConcentrationLXe)*self.RelativeVolatility*self.CondensationProb
        VaporizationTerm *= 3600.*24. # to kg*ppb/day

        # Condenzation
        CondensationTerm = RelativeFactor*(self.StableCoolingPower/self.LatentHeatXenon)*(PreviousConcentrationGXe)*self.CondensationProb
        CondensationTerm *= 3600.*24. # to kg*ppb/day

        # Condenzation from flow with cooling power base
        FlowCondensationTerm = 0 
        FlowCondensationTerm = IsFlowCondense*np.fmax(0,(CoolingPower - self.StableCoolingPower)/self.LatentHeatXenon)*(PreviousConcentrationGXe)*self.CondensationProb
        FlowCondensationTerm *= 3600.*24. # to kg*ppb/day

        # Back Flow during getter deficiency period
        GXeBackFlowTerm=0
        LXeBackFlowTerm=0
        if(LiquidFlow+GasFlow>0):
            MixedConcentration  = ((GasFlowTerm+LiquidFlowTerm))/((LiquidFlow+GasFlow)*self.StandardXenonDensity*1e-3*60.*24.)
            BackConcentration  = MixedConcentration * BackFraction
            LXeBackFlowTerm =  (BackConcentration/(HEeff + self.RelativeVolatility*(1 - HEeff)))*HEeff*(LiquidFlow+GasFlow-BellFlow)
            LXeBackFlowTerm *= self.StandardXenonDensity*1e-3*60.*24.
            GXeBackFlowTerm =  (self.RelativeVolatility*BackConcentration/(HEeff + self.RelativeVolatility*(1 - HEeff)))*(1-HEeff)*(LiquidFlow+GasFlow-BellFlow)+BackConcentration*BellFlow 
            GXeBackFlowTerm *= self.StandardXenonDensity*1e-3*60.*24.

        # concentratin change
        ConcentrationChangeGXe += -GasFlowTerm
        ConcentrationChangeGXe += VaporizationTerm
        ConcentrationChangeGXe += -CondensationTerm
        ConcentrationChangeGXe += GXeBackFlowTerm
        ConcentrationChangeGXe += -FlowCondensationTerm

        ConcentrationChangeLXe += -LiquidFlowTerm
        ConcentrationChangeLXe += -VaporizationTerm
        ConcentrationChangeLXe += CondensationTerm
        ConcentrationChangeLXe += LXeBackFlowTerm
        ConcentrationChangeLXe += FlowCondensationTerm

        #Mass normalization
        ConcentrationChangeGXe /= self.MassGXe
        ConcentrationChangeLXe /= self.MassLXe

        #time step
        ConcentrationChanges[0] = ConcentrationChangeGXe
        ConcentrationChanges[1] = ConcentrationChangeLXe
        return ConcentrationChanges

    def CalculateImpurityConcentration(self): #RK
        cdef:
            float TrueTimeStep, PreviousConcentrationGXe, PreviousConcentrationLXe,day
            int index_change
            float ImpuritySpike

        # main function for calculating the impurity trend
        ConcentrationsGXe = np.zeros(int(self.Npoints/2)+1,dtype=float)
        ConcentrationsLXe = np.zeros(int(self.Npoints/2)+1,dtype=float)
        CalculatedDays  = np.zeros(int(self.Npoints/2)+1,dtype=float) #= []
        k1 =np.zeros(self.NInt,dtype=float)
        k2 =np.zeros(self.NInt,dtype=float)
        k3 =np.zeros(self.NInt,dtype=float)
        k4 =np.zeros(self.NInt,dtype=float)
        PreviousConcentrations=np.zeros(self.NInt,dtype=float)
        # Previous concentrations
        PreviousConcentrations[0] = self.InitialConcentrationLXe *self.RelativeVolatility
        PreviousConcentrations[1] = self.InitialConcentrationLXe 
        ConcentrationsGXe[0] = PreviousConcentrations[0]
        ConcentrationsLXe[0] = PreviousConcentrations[1]
        CalculatedDays[0] = self.GlobalDays[0]
        for i in range(2,self.Npoints,2):
            TrueTimeStep=self.GlobalDays[i] - self.GlobalDays[i-2]
            if np.fabs((self.GlobalDays[i] - self.GlobalDays[i-1])/(self.GlobalDays[i-1] - self.GlobalDays[i-2])-1) > 0.02:
                print("Error on the timestep more than 2%.", self.GlobalDays[i])

            # check and add, if need to consider sudden drops from previous period
            ImpuritySpikeGXe=self.GlobalDropGXe[i]
            ImpuritySpikeLXe=self.GlobalDropLXe[i]

            #Add InpuritySpike for Both
            PreviousConcentrations[0] += ImpuritySpikeGXe / self.MassGXe * 0.133 * 1e9
            PreviousConcentrations[1] += ImpuritySpikeLXe / self.MassLXe * 0.133 * 1e9

            # Calculate based on R-K
            k1=self.GetChangeOfImpurityConcentration(i-2,PreviousConcentrations)
            k2=self.GetChangeOfImpurityConcentration(i-1,PreviousConcentrations + k1*TrueTimeStep/2.)
            k3=self.GetChangeOfImpurityConcentration(i-1,PreviousConcentrations + k2*TrueTimeStep/2.)
            k4=self.GetChangeOfImpurityConcentration(i,PreviousConcentrations + k3*TrueTimeStep)

            # Add changes
            ConcentrationChanges = np.zeros(self.NInt,dtype=float)
            ConcentrationChanges += (k1 + 2*k2+ 2*k3+ k4)*TrueTimeStep/6.
            if not (np.all(np.asarray(PreviousConcentrations)+ConcentrationChanges>0)):
                ConcentrationsGXe = np.asarray([PreviousConcentrations[0],PreviousConcentrations[0]])
                ConcentrationsLXe =np.asarray([PreviousConcentrations[1],PreviousConcentrations[1]])
                CalculatedDays=np.asarray([self.GlobalDays[0],self.GlobalDays[-1]])
                break
            PreviousConcentrations += ConcentrationChanges 

            # append
            index = int(i/2)
            ConcentrationsGXe[index] = PreviousConcentrations[0]
            ConcentrationsLXe[index] = PreviousConcentrations[1]
            CalculatedDays[index]=self.GlobalDays[i]

        # Interpolate after the calculation
        self.inter_ConcentrationsGXe = interp1d(CalculatedDays, ConcentrationsGXe, bounds_error=False, fill_value = 'extrapolate')
        self.inter_ConcentrationsLXe = interp1d(CalculatedDays, ConcentrationsLXe, bounds_error=False, fill_value = 'extrapolate')
        return

    def GetConcentrations(self, unixtimes):
        return [self.GetGXeConcentration(unixtimes),
                     self.GetLXeConcentration(unixtimes),self.GetLXeConcentration(unixtimes),]

    def GetConcentrationsAve(self, unixtimes, Error):
        return [self.GetGXeConcentration(unixtimes),
                     self.GetLXeConcentration(unixtimes),self.GetLXeConcentration(unixtimes)]

    def GetGXeConcentrationAve(self, unixtimes,Error):
        EffDays = (np.asarray(unixtimes) - self.MinUnixTime) / 3600. / 24.#self.HistorianData.GetReferenceUnixTime()
        ErrDays = np.asarray(Error)/ 3600. / 24.
        C0 = self.inter_ConcentrationsGXe(EffDays)
        C1 = self.inter_ConcentrationsGXe(EffDays+ErrDays/2.)
        C2 = self.inter_ConcentrationsGXe(EffDays+ErrDays)
        C3 = self.inter_ConcentrationsGXe(EffDays-ErrDays/2.)
        C4 = self.inter_ConcentrationsGXe(EffDays-ErrDays)

        return (C0+C1+C2+C3+C4)/5.

    def GetGXeConcentration(self, unixtimes):
        EffDays = (np.asarray(unixtimes) - self.MinUnixTime) / 3600. / 24.#self.HistorianData.GetReferenceUnixTime()
        return self.inter_ConcentrationsGXe(EffDays)

    def GetFlows(self, unixtimes):
        EffDays = (np.asarray(unixtimes) - self.MinUnixTime) / 3600. / 24.#self.HistorianData.GetReferenceUnixTime()
        g=interp1d(self.GlobalDays, self.GlobalGasFlows, fill_value = 'extrapolate')
        l=interp1d(self.GlobalDays, self.GlobalLiquidFlows, fill_value = 'extrapolate')
        return g(EffDays),l(EffDays) 


    def GetLXeConcentration(self, unixtimes):
        EffDays = (np.asarray(unixtimes) - self.MinUnixTime) / 3600. / 24.#self.HistorianData.GetReferenceUnixTime()
        return self.inter_ConcentrationsLXe(EffDays)                       
                                                                           
    def GetLXeConcentrationAve(self, unixtimes,Error):                     
        EffDays = (np.asarray(unixtimes) - self.MinUnixTime) / 3600. / 24.#self.HistorianData.GetReferenceUnixTime()
        ErrDays = np.asarray(Error)/ 3600. / 24.
        C0 = self.inter_ConcentrationsLXe(EffDays)
        C1 = self.inter_ConcentrationsLXe(EffDays+ErrDays/2.)
        C2 = self.inter_ConcentrationsLXe(EffDays+ErrDays)
        C3 = self.inter_ConcentrationsLXe(EffDays-ErrDays/2.)
        C4 = self.inter_ConcentrationsLXe(EffDays-ErrDays)
        return (C0+C1+C2+C3+C4)/5.


    def GetPhaseEx(self, unixtimes):
        IsFlowCondense = self.GlobalFlowCondense
        #LowFlow = self.GlobalLowFlow
        LiquidFlow = self.GlobalLiquidFlows
        GasFlow= self.GlobalGasFlows
        BellFlow= self.GlobalBellFlows
        CoolingPower= self.GlobalCoolingPowers
        BackFraction = self.GlobalBackFractions
        InnerVesselTemp=self.GlobalInnerVesselTemps
        BelowBellTemp=self.GlobalBelowBellTemps
        GXecirc = self.GlobalGXecirc
        HEeff = self.GlobalHEeff
        Ig, Il,LN2 = self.GetConcentrations(self.GlobalUnixTimes)

        # Vaporization
        VaporizationTerm = self.RelativeCondense*(self.StableCoolingPower/self.LatentHeatXenon)*(Il)*self.RelativeVolatility*self.CondensationProb
        VaporizationTerm *= 3600.*24. # to kg*ppb/day

        # Condenzation
        CondensationTerm = self.RelativeCondense*(self.StableCoolingPower/self.LatentHeatXenon)*(Ig)*self.CondensationProb
        CondensationTerm *= 3600.*24. # to kg*ppb/day

        # Condenzation from flow
        FlowCondensationTerm = IsFlowCondense*np.fmax(0,(GasFlow+LiquidFlow-BellFlow)*(1-HEeff)+BellFlow-GasFlow)*self.StandardXenonDensity*(Ig)*self.CondensationProb
        FlowCondensationTerm *= 1e-3*60.*24. # to kg*ppb/day

        inter_Vapo = interp1d(self.GlobalUnixTimes, VaporizationTerm, bounds_error=False, fill_value = 'extrapolate')
        inter_Cond = interp1d(self.GlobalUnixTimes, CondensationTerm, bounds_error=False, fill_value = 'extrapolate')
        inter_FlowCond = interp1d(self.GlobalUnixTimes, FlowCondensationTerm, bounds_error=False, fill_value = 'extrapolate')
        return inter_Vapo(unixtimes), inter_Cond(unixtimes),inter_FlowCond(unixtimes) 

    def GetOutgassing(self, unixtimes):
        LXeOutgassing = self.GetLXeOutgassingRate()
        GXeOutgassing = self.GetGXeOutgassingRate()
        GXeBackFlowTerm=np.zeros(len(LXeOutgassing))
        LXeBackFlowTerm=np.zeros(len(LXeOutgassing))
        indices = np.where(self.GlobalTotFlows>0)
        inter_LXeOutgassing = interp1d(self.GlobalUnixTimes, LXeOutgassing, bounds_error=False, fill_value = 'extrapolate')
        inter_GXeOutgassing = interp1d(self.GlobalUnixTimes, GXeOutgassing, bounds_error=False, fill_value = 'extrapolate')

        return inter_GXeOutgassing(unixtimes),inter_LXeOutgassing(unixtimes)



    def GetStandardXenonDensity(self):
        return self.StandardXenonDensity

    def GetStableCoolingPower(self):
        return self.StableCoolingPower

    def GetLatentHeat(self):
        return self.LatentHeatXenon

    def GetGXeMass(self):
        return self.MassGXe

    def GetLXeMass(self):
        return self.MassLXe

    def GetDefaultTimeStep(self):
        return self.DefaultTimeStep

    def GetAttachingRateCorrectionFactor(self, unixtimes, SysErr):
        CathodeVoltages = self.HistorianData.GetCathode(unixtimes)
        Field = CathodeVoltages / 100. 
        return (self.AttachingRateAsField (Field) + SysErr*self.AttachingRateErrorAsField (Field)) / self.AttachingRateAsField (self.ReferenceField)

    # new @ 2016-11-16
    def ManualChangingFlows(self, NewLiquidFlow, NewGasFlow, start_unixtime, end_unixtime):
        self.HistorianData.AddGasOnlyPeriod([start_unixtime, end_unixtime])
        for i, day in enumerate(self.GlobalDays):
            unixtime = day * 24. * 3600.+ self.MinUnixTime#self.HistorianData.GetReferenceUnixTime()
            if unixtime<start_unixtime or unixtime>end_unixtime:
                continue
            self.GlobalLiquidFlows[i] = NewLiquidFlow
            self.GlobalGasFlows[i] = NewGasFlow
        return

