#cython: language_level=3
import re
import numpy as np
import scipy as sp
import os, sys
import pickle
import pandas as pd
from scipy.interpolate import interp1d

import FormPars
from FormPars import *
import MCMC_Tools
import more_itertools as miter
import copy


#import ROOT
#from ROOT import TFile
#from ROOT import TTree
#from ROOT import TBranch

class MyHistorianData:

    def __init__(self, SCPickleFile):
        self.FormPars = MyFormPars()
        self.ReferenceUnixTime = 0
        self.SpecialPeriod = self.FormPars.GetSpecialPeriods()
        # during the gas only period, the circulation route is very different
        # see: https://xecluster.lngs.infn.it/dokuwiki/lib/exe/fetch.php?media=xenon:xenon1t:org:commissioning:meetings:160817:xenon1t-operating_modes-circulation-mode_3_x-circulation_gxe_high_flow-v0.pdf
        # so now the total flow is not basically the sum of FCV201&FCV202
        # but only FCV201
        self.GasOnlyPeriod = self.FormPars.GetGasOnlyPeriods()
        self.GetterDeficiencyConfigs = []
        self.HeatingTrends = self.FormPars.GetHeatingTrends() # W
        if not self.LoadFile(SCPickleFile):
            raise ValueError("SC pickle file error!")
        self.MaximumHeatingPower = self.FormPars.GetMaximumHeatingPower() # W
        return

    def AddGasOnlyPeriod(self, Period):
        self.GasOnlyPeriod.append(Period)
        return

    def AddOneGetterDeficiencyConfig(self, Config):
        if len(Config)!=3:
            print("The config doesn't match the format! Nothing done!")
            return
        self.GetterDeficiencyConfigs.append(Config)
        return

    def PopOneGetterDeficiencyConfig(self):
        if len(self.GetterDeficiencyConfigs)>0:
            self.GetterDeficiencyConfigs.pop()
        return

    def GetGetterDeficiencyConfigs(self):
        return self.GetterDeficiencyConfigs

    def __eq__(self, other):
        # general ones
        self.ReferenceUnixTime = other.ReferenceUnixTime
        self.SpecialPeriod = other.SpecialPeriod
        self.GasOnlyPeriod = other.GasOnlyPeriod
        self.GetterDeficiencyConfigs = other.GetterDeficiencyConfigs
        self.MaximumHeatingPower = other.MaximumHeatingPower
        self.ProbableCathodeRange = other.ProbableCathodeRange
        # Min unixtimes
        self.MinUnixTime_FC201 = other.MinUnixTime_FC201
        self.MinUnixTime_FC202 = other.MinUnixTime_FC202
        self.MinUnixTime_FCV101 = other.MinUnixTime_FCV101
        self.MinUnixTime_FCV102 = other.MinUnixTime_FCV102
        self.MinUnixTime_FCV103 = other.MinUnixTime_FCV103
        self.MinUnixTime_FCV104 = other.MinUnixTime_FCV104
        self.MinUnixTime_FIC401 = other.MinUnixTime_FIC401
        self.MinUnixTime_HeatPower = other.MinUnixTime_HeatPower
        self.MinUnixTime_FV217 = other.MinUnixTime_FV217
        self.MinUnixTime_FV224 = other.MinUnixTime_FV224
        self.MinUnixTime_CRYTE104 = other.MinUnixTime_CRYTE104
        self.MinUnixTime_CRYTE107 = other.MinUnixTime_CRYTE107
        # Max unixtimes
        self.MaxUnixTime_FC201 = other.MaxUnixTime_FC201
        self.MaxUnixTime_FC202 = other.MaxUnixTime_FC202
        self.MaxUnixTime_FCV101 = other.MaxUnixTime_FCV101
        self.MaxUnixTime_FCV102 = other.MaxUnixTime_FCV102
        self.MaxUnixTime_FCV103 = other.MaxUnixTime_FCV103
        self.MaxUnixTime_FCV104 = other.MaxUnixTime_FCV104
        self.MaxUnixTime_FIC401 = other.MaxUnixTime_FIC401
        self.MaxUnixTime_HeatPower = other.MaxUnixTime_HeatPower
        self.MaxUnixTime_FV217 = other.MaxUnixTime_FV217
        self.MaxUnixTime_FV224 = other.MaxUnixTime_FV224
        self.MaxUnixTime_CRYTE104 = other.MaxUnixTime_CRYTE104
        self.MaxUnixTime_CRYTE107 = other.MaxUnixTime_CRYTE107
        # interpolation
        self.inter_FC201 = other.inter_FC201
        self.inter_FC202 = other.inter_FC202
        self.inter_FCV101 = other.inter_FCV101
        self.inter_FCV102 = other.inter_FCV102
        self.inter_FCV103 = other.inter_FCV103
        self.inter_FCV104 = other.inter_FCV104
        self.inter_FIC401 = other.inter_FIC401
        self.inter_HeatPower = other.inter_HeatPower
        self.inter_FV217 = other.inter_FV217
        self.inter_FV224 = other.inter_FV224
        self.inter_CRYTE104 = other.inter_CRYTE104
        self.inter_CRYTE107 = other.inter_CRYTE107
        self.MaximumUnixTime = other.MaximumUnixTime
        return
        

    def LoadFile(self, Filename):
        print("===== start loading SC pickle =======")
        PickleData = pickle.load( open(Filename, 'rb'))
        dict_FC201 = PickleData["PUR_FC201"]
        dict_FC202 = PickleData["PUR_FC202"]
        dict_FCV101 = PickleData["CRY_FCV101"]
        dict_FCV102 = PickleData["CRY_FCV102"]
        dict_FCV103 = PickleData["CRY_FCV103"]
        dict_FCV104 = PickleData["CRY_FCV104"]
        dict_FIC401 = PickleData["DST_FIC401"]
        dict_HeatPower = PickleData["CRY_R121P"]
        dict_FV217 = PickleData["PUR_FV217V"]
        dict_FV224 = PickleData["PUR_FV224V"]
        dict_Cathode = PickleData["TPC_Monitor_Voltage"]
        dict_CRYTE104 = PickleData["CRY_TE104"]
        dict_CRYTE107 = PickleData["CRY_TE107"]
        if not dict_FC201:
            raise ValueError("FC 201 not available")
        if not dict_FC202:
            raise ValueError("FC 202 not available")
        if not dict_FCV101:
            raise ValueError("FCV 101 not available")
        if not dict_FCV102:
            raise ValueError("FCV 102 not available")
        if not dict_FCV103:
            raise ValueError("FCV 103 not available")
        if not dict_FCV104:
            raise ValueError("FCV 104 not available")
        if not dict_FIC401:
            raise ValueError("FIC 401s not available")
        if not dict_HeatPower:
            raise ValueError("Heat power not available")
        if not dict_FV217:
            raise ValueError("FV 217 not available")
        if not dict_FV224:
            raise ValueError("FV 224 not available")
        if not dict_Cathode:
            raise ValueError("Cathode not available")
        if not dict_CRYTE104:
            raise ValueError("CRY TE104 not available")
        if not dict_CRYTE107:
            raise ValueError("CRY TE107 not available")
        self.MinUnixTime_FC201, self.MaxUnixTime_FC201, self.inter_FC201 = self.GetInterpolationLiquidFlow(dict_FC201)
        self.MinUnixTime_FC202, self.MaxUnixTime_FC202,  self.inter_FC202 = self.GetInterpolationLiquidFlow(dict_FC202)
        self.MinUnixTime_FCV101, self.MaxUnixTime_FCV101, self.inter_FCV101 = self.GetInterpolation(dict_FCV101)
        self.MinUnixTime_FCV102, self.MaxUnixTime_FCV102, self.inter_FCV102 = self.GetInterpolation(dict_FCV102)
        self.MinUnixTime_FCV103, self.MaxUnixTime_FCV103, self.inter_FCV103 = self.GetInterpolation(dict_FCV103)
        self.MinUnixTime_FCV104, self.MaxUnixTime_FCV104, self.inter_FCV104 = self.GetInterpolation(dict_FCV104)
        self.MinUnixTime_FIC401, self.MaxUnixTime_FIC401, self.inter_FIC401 = self.GetInterpolation(dict_FIC401)
        self.MinUnixTime_HeatPower, self.MaxUnixTime_HeatPower, self.inter_HeatPower, self.inter_HeaterCorrection = self.GetInterpolationHeater(dict_HeatPower)
        self.MinUnixTime_FV217, self.MaxUnixTime_FV217, self.inter_FV217 = self.GetInterpolationSpecial(dict_FV217)
        self.MinUnixTime_FV224, self.MaxUnixTime_FV224, self.inter_FV224 = self.GetInterpolationSpecial(dict_FV224)
        self.MinUnixTime_Cathode, self.MaxUnixTime_Cathode, self.inter_Cathode = self.GetInterpolationCathode(dict_Cathode)
        self.MinUnixTime_CRYTE104, self.MaxUnixTime_CRYTE104, self.inter_CRYTE104 = self.GetInterpolationTempCelsius(dict_CRYTE104)
        self.MinUnixTime_CRYTE107, self.MaxUnixTime_CRYTE107, self.inter_CRYTE107 = self.GetInterpolationTempCelsius(dict_CRYTE107)

        self.ReferenceUnixTime = MCMC_Tools.GetUnixTimeFromTimeStamp(self.FormPars.GetMinTimeStamp())
        self.MaximumUnixTime = np.max([self.MaxUnixTime_FC201,
                                                                      self.MaxUnixTime_FC202,
                                                                      self.MaxUnixTime_FCV101,
                                                                      self.MaxUnixTime_FCV102,
                                                                      self.MaxUnixTime_FCV103,
                                                                      self.MaxUnixTime_FCV104,
                                                                      self.MaxUnixTime_FIC401,
                                                                      self.MaxUnixTime_HeatPower,
                                                                      self.MaxUnixTime_FV217,
                                                                      self.MaxUnixTime_FV224,
                                                                      self.MaxUnixTime_Cathode,
                                                                      self.MaxUnixTime_CRYTE104,
                                                                      self.MaxUnixTime_CRYTE107,
                                                                    ], axis=0
                                                                   )
        print("===== finish loading SC pickle =======")
        return True

    def GetReferenceUnixTime(self):
        return self.ReferenceUnixTime

    def GetMaximumUnixTime(self):
        return self.MaximumUnixTime

    def GetInterpolation(self, Dict):
        # get the interp1d from a tree
        UnixTimes = Dict['unixtimes']
        Values = Dict['values']
        MinUnixTime = min(UnixTimes)
        MaxUnixTime = max(UnixTimes)
        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, Values))

    def GetInterpolationLiquidFlow(self, Dict):
        # get the interp1d from a tree
        UnixTimes = np.asarray(Dict['unixtimes'])
        Values = np.asarray(Dict['values'])

        T_MagPump =  1528813800 #Magpump install preparation Jun 05 - Jun 12
        NChunk = int(20)
        UnixTimes_QDrive = copy.deepcopy(UnixTimes)
        Values_QDrive =  copy.deepcopy(Values)
        UnixTimes_MagPump = copy.deepcopy(UnixTimes)
        Values_MagPump =  copy.deepcopy(Values)
        Values_QDrive =  Values_QDrive[np.where(UnixTimes_QDrive<T_MagPump)]
        UnixTimes_QDrive = UnixTimes_QDrive[np.where(UnixTimes_QDrive<T_MagPump)]
        Values_MagPump =  Values_MagPump[np.where(UnixTimes_MagPump>=T_MagPump)]
        UnixTimes_MagPump = UnixTimes_MagPump[np.where(UnixTimes_MagPump>=T_MagPump)]

        vlen10=len(UnixTimes_MagPump)-len(UnixTimes_MagPump)%NChunk
        UnixTimes_MagPump=UnixTimes_MagPump[0:vlen10]
        Values_MagPump=Values_MagPump[0:vlen10]
        chunked=np.asarray(list(miter.chunked(Values_MagPump,NChunk)))
        t_chunked=np.asarray(list(miter.chunked(UnixTimes_MagPump,NChunk)))
        Values_MagPump=np.average(chunked,axis=1)
        UnixTimes_MagPump=np.average(t_chunked,axis=1)

        UnixTimes = np.append(UnixTimes_QDrive,UnixTimes_MagPump)
        Values = np.append(Values_QDrive,Values_MagPump)
        MinUnixTime = min(UnixTimes)
        MaxUnixTime = max(UnixTimes)
        
        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, Values))




    def GetInterpolationHeater(self, Dict):
        # get the interp1d from a tree
        # cooling power decreases over time so CRY_R121P decreases as well
        # needs to be accounted for when returning cooling power of GXe
        UnixTimes = Dict['unixtimes']
        Values = Dict['values']
        MinUnixTime = min(UnixTimes)
        MaxUnixTime = max(UnixTimes)
        firstDecrease = 0
        unixtimeChanges = []
        valueChanges = []
        for i,(u,v) in enumerate(self.HeatingTrends):
            unixtimeChanges.append(u[0])
            unixtimeChanges.append(u[1] - 1)
            if v[1] != 0 and firstDecrease == 0:
                firstDecrease = v[1]
            valueChanges.append(v[1] - firstDecrease)
            valueChanges.append(v[0] * (u[1] - u[0]) + v[1] - firstDecrease)
        indicesToKeep = np.where((UnixTimes < 1526820000) & (UnixTimes > 1525860000))[0]
        for i in indicesToKeep:
            #print(i, Values[i])
            Values[i] =39
        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, Values), interp1d(unixtimeChanges, valueChanges))


    def GetInterpolationCathode(self, Dict):
        UnixTimes = Dict['unixtimes']
        Values = Dict['values']
        MinUnixTime = min(UnixTimes)
        MaxUnixTime = max(UnixTimes)
        ReturnValues = []
        previousValue = 15. #default
        CathodeVoltages = self.FormPars.GetCathodeVoltages()
        for unixtime, value in zip(UnixTimes, Values):
            for Array in CathodeVoltages:
                if unixtime >= Array[0][0] and unixtime < Array[0][1]:
                    # force voltage to take on value
                    if Array[1][0] == Array[1][1]:
                        value = Array[1][0]
                        ReturnValues.append(value)
                        previousValue = value
                    else:
                        if value >= Array[1][0] and value < Array[1][1]:
                            ReturnValues.append(value)
                            previousValue = value
                        else:
                            ReturnValues.append(previousValue)
                    continue

        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, ReturnValues, bounds_error=False, fill_value=(ReturnValues[0], ReturnValues[-1])))

    # principle
    # Two conditions that consider there's a valve status changing
    # 1) 0 -> 1 or 1->0 changing
    #     putting another '0' (or '1') 1 second before '1' (or '0')
    # 2) 1 <-> 1 longer than 10hr
    #     putting 4hr '1' before/after the end/begin '1'
    def GetInterpolationSpecial(self, Dict):
        # get the interp1d from a tree
        UnixTimes = []
        Values = []
        nEvents = len(Dict['unixtimes'])
        MinUnixTime = 10000000000
        MaxUnixTime = 0
        # generate the first one
        unixtime = Dict['unixtimes'][0]
        UnixTimes.append(unixtime-1)
        Values.append(0) # default starting point zero
        if unixtime < MinUnixTime:
            MinUnixTime = unixtime
        if unixtime > MaxUnixTime:
            MaxUnixTime = unixtime
        previous_value = 0
        previous_unixtime = unixtime - 1
        for event_id in range(nEvents):
            unixtime = Dict['unixtimes'][event_id]
            value = float(Dict['values'][event_id])
            # two choices
            if not value==previous_value:
                UnixTimes.append(unixtime-1)
                Values.append(previous_value)
                UnixTimes.append(unixtime)
                Values.append(value)
            elif ( (unixtime>lower and unixtime<upper) for (lower, upper) in self.SpecialPeriod):
                UnixTimes.append(unixtime)
                Values.append(value)
            elif (unixtime-previous_unixtime)/3600. > 10 and value==previous_value and value==1:
                UnixTimes.append(previous_unixtime+4.*3600.)
                Values.append(previous_value)
                UnixTimes.append(previous_unixtime+4.*3600.+1)
                Values.append(abs(1-previous_value))
                UnixTimes.append(unixtime - 4.*3600-1)
                Values.append(abs(1-value))
                UnixTimes.append(unixtime - 4.*3600)
                Values.append(value)
                UnixTimes.append(unixtime)
                Values.append(value)
            else:
                UnixTimes.append(unixtime)
                Values.append(value)
            previous_unixtime = unixtime
            previous_value = value
            if unixtime < MinUnixTime:
                MinUnixTime = unixtime
            if unixtime > MaxUnixTime:
                MaxUnixTime = unixtime
        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, Values))


    def GetInterpolationTempCelsius(self, Dict):
        # convert to Kelvin
        UnixTimes = Dict['unixtimes']
        Values = Dict['values'] + 273.15
        MinUnixTime = min(UnixTimes)
        MaxUnixTime = max(UnixTimes)
        return (MinUnixTime, MaxUnixTime, interp1d(UnixTimes, Values))


    # return (liquid flow, gas flow, cooling power, cathode voltage)
    def GetHistorian(self, unixtime):
        GasFlow = self.GetGasFlow(unixtime)
        TotalFlow, ByPassFraction, DSTFlow = self.GetLiquidFlow(unixtime)
        BellFlow = self.GetBellFlow(unixtime)
        BellFlow = np.fmax(BellFlow-0.05,0)
        TrueGasFlow = GasFlow*(1. - ByPassFraction)
        TrueLiquidFlow = (TotalFlow-GasFlow)*(1.-ByPassFraction)
        if(unixtime<self.GasOnlyPeriod[0][0]):
            TrueLiquidFlow = (TotalFlow-GasFlow-DSTFlow)*(1.-ByPassFraction)
        TrueLiquidFlow = np.fmax(TrueLiquidFlow-0.7,0)
        TrueGasFlow = np.fmax(TrueGasFlow-0.03,0)
        CathodeVoltage = self.GetCathodeVoltage(unixtime)
        BelowBellTemp = self.GetBelowBellTemp(unixtime)
        InnerVesselTemp = self.GetInnerVesselTemp(unixtime)
        # if it is gas only
        for Period in self.GasOnlyPeriod:
            if unixtime>Period[0] and unixtime<Period[1]:
                return (
                             0., # not liquid flow
                             TotalFlow*(1. - ByPassFraction), # note here it is not 100% correct if we decide not bypassing getter 202
                             BellFlow,
                             self.GetCoolingPower(unixtime),
                             CathodeVoltage,
                             BelowBellTemp,
                             InnerVesselTemp,
                            )
        # if it is in period with suspecious getter deficiency
        return (
                     TrueLiquidFlow,
                     TrueGasFlow,
                     BellFlow,
                     self.GetCoolingPower(unixtime),
                     CathodeVoltage,
                     BelowBellTemp,
                     InnerVesselTemp,
                    )

    # return valve status
    def GetValveStatus(self, unixtime):
        EffUnixTime_FV217 = unixtime
        if unixtime<self.MinUnixTime_FV217:
            EffUnixTime_FV217 = self.MinUnixTime_FV217
        if unixtime>self.MaxUnixTime_FV217:
            EffUnixTime_FV217 = self.MaxUnixTime_FV217
        EffUnixTime_FV224 = unixtime
        if unixtime<self.MinUnixTime_FV224:
            EffUnixTime_FV224 = self.MinUnixTime_FV224
        if unixtime>self.MaxUnixTime_FV224:
            EffUnixTime_FV224 = self.MaxUnixTime_FV224
        FV217_Status = self.inter_FV217(EffUnixTime_FV217)
        if FV217_Status<0.5:
            FV217_Status=0
        else:
            FV217_Status=1
        FV224_Status = self.inter_FV224(EffUnixTime_FV224)
        if FV224_Status<0.5:
            FV224_Status=0
        else:
            FV224_Status=1
        return (FV217_Status, FV224_Status)

    def GetBellFlow(self, unixtime):
        EffUnixTime_FCV104 = unixtime
        if unixtime<self.MinUnixTime_FCV104:
            EffUnixTime_FCV104 = self.MinUnixTime_FCV104
        if unixtime>self.MaxUnixTime_FCV104:
            EffUnixTime_FCV104 = self.MaxUnixTime_FCV104
        TotalFlow = self.inter_FCV104(EffUnixTime_FCV104)
        if TotalFlow<=0:
            return 0
        return TotalFlow


    # return liquid flow
    def GetLiquidFlow(self, unixtime):
        EffUnixTime_FC201 = unixtime
        if unixtime<self.MinUnixTime_FC201:
            EffUnixTime_FC201 = self.MinUnixTime_FC201
        if unixtime>self.MaxUnixTime_FC201:
            EffUnixTime_FC201 = self.MaxUnixTime_FC201
        EffUnixTime_FC202 = unixtime
        if unixtime<self.MinUnixTime_FC202:
            EffUnixTime_FC202 = self.MinUnixTime_FC202
        if unixtime>self.MaxUnixTime_FC202:
            EffUnixTime_FC202 = self.MaxUnixTime_FC202
        EffUnixTime_FIC401 = unixtime
        if unixtime<self.MinUnixTime_FIC401:
            EffUnixTime_FIC401 = self.MinUnixTime_FIC401
        if unixtime>self.MaxUnixTime_FIC401:
            EffUnixTime_FIC401 = self.MaxUnixTime_FIC401
        ByPass_FC201, ByPass_FC202 = self.GetValveStatus(unixtime)
        TotalFlow = self.inter_FC201(EffUnixTime_FC201) + self.inter_FC202(EffUnixTime_FC202)
        EffFlow = self.inter_FC201(EffUnixTime_FC201)*(1-ByPass_FC201)+self.inter_FC202(EffUnixTime_FC202)*(1-ByPass_FC202)
        DSTFlow = self.inter_FIC401(EffUnixTime_FIC401)
        if EffFlow<0 or TotalFlow<=0:
            return (0, 0, 0)
        ByPassFraction = (TotalFlow - EffFlow) / TotalFlow
        return (TotalFlow, ByPassFraction, DSTFlow)

    def GetFCV201Flow(self, unixtime):
        EffUnixTime_FC201 = unixtime
        if unixtime<self.MinUnixTime_FC201:
            EffUnixTime_FC201 = self.MinUnixTime_FC201
        if unixtime>self.MaxUnixTime_FC201:
            EffUnixTime_FC201 = self.MaxUnixTime_FC201
        return self.inter_FC201(EffUnixTime_FC201)

    # return gas flow
    def GetGasFlow(self, unixtime):
        EffUnixTime_FCV101 = unixtime
        if unixtime<self.MinUnixTime_FCV101:
            EffUnixTime_FCV101 = self.MinUnixTime_FCV101
        if unixtime>self.MaxUnixTime_FCV101:
            EffUnixTime_FCV101 = self.MaxUnixTime_FCV101
        EffUnixTime_FCV102 = unixtime
        if unixtime<self.MinUnixTime_FCV102:
            EffUnixTime_FCV102 = self.MinUnixTime_FCV102
        if unixtime>self.MaxUnixTime_FCV102:
            EffUnixTime_FCV102 = self.MaxUnixTime_FCV102
        EffUnixTime_FCV103 = unixtime
        if unixtime<self.MinUnixTime_FCV103:
            EffUnixTime_FCV103 = self.MinUnixTime_FCV103
        if unixtime>self.MaxUnixTime_FCV103:
            EffUnixTime_FCV103 = self.MaxUnixTime_FCV103
        Flow = self.inter_FCV101(EffUnixTime_FCV101)+self.inter_FCV102(EffUnixTime_FCV102)+self.inter_FCV103(EffUnixTime_FCV103)
        if Flow<0:
            return 0
        return Flow

    # return the cooling power
    def GetUncorrectedCoolingPower(self, unixtime):
        EffUnixTime_CoolingPower = unixtime
        if unixtime<self.MinUnixTime_HeatPower:
            EffUnixTime_CoolingPower = self.MinUnixTime_HeatPower
        if unixtime>self.MaxUnixTime_HeatPower:
            EffUnixTime_CoolingPower = self.MaxUnixTime_HeatPower
        return self.MaximumHeatingPower - self.inter_HeatPower(EffUnixTime_CoolingPower)


    def GetCoolingPower(self, unixtime):
        # GXe cooling power after correcting for cooling tower decrease
        EffUnixTime_CoolingPower = unixtime
        if unixtime<self.MinUnixTime_HeatPower:
            EffUnixTime_CoolingPower = self.MinUnixTime_HeatPower
        if unixtime>self.MaxUnixTime_HeatPower:
            EffUnixTime_CoolingPower = self.MaxUnixTime_HeatPower
        return self.MaximumHeatingPower - self.inter_HeatPower(EffUnixTime_CoolingPower) + self.inter_HeaterCorrection(EffUnixTime_CoolingPower)


    # get the end point liquid flow
    def GetEndLiquidFlow(self):
        EffUnixTime_FC201 = self.MaxUnixTime_FC201
        EffUnixTime_FC202 = self.MaxUnixTime_FC202
        return self.inter_FC201(EffUnixTime_FC201)+self.inter_FC202(EffUnixTime_FC202)

    # get the cathode voltage
    def GetCathodeVoltage(self, unixtime):
        EffUnixTime_Cathode = unixtime
        if unixtime<self.MinUnixTime_Cathode:
            EffUnixTime_Cathode = self.MinUnixTime_Cathode
        if unixtime>self.MaxUnixTime_Cathode:
            EffUnixTime_Cathode = self.MaxUnixTime_Cathode
        return self.inter_Cathode (EffUnixTime_Cathode)

    def GetCathode(self, unixtimes):
        return self.inter_Cathode (unixtimes)

    # get cryostat below bell temperature (LXe)
    def GetBelowBellTemp(self, unixtime):
        EffUnixTime_BBTemp = unixtime
        if unixtime<self.MinUnixTime_CRYTE104:
            EffUnixTime_BBTemp = self.MinUnixTime_CRYTE104
        if unixtime>self.MaxUnixTime_CRYTE104:
            EffUnixTime_BBTemp = self.MaxUnixTime_CRYTE104
        return self.inter_CRYTE104 (EffUnixTime_BBTemp)

    # get PTR-101 tower inner veseel temperature
    def GetInnerVesselTemp(self, unixtime):
        EffUnixTime_InnerVessel = unixtime
        if unixtime<self.MinUnixTime_CRYTE107:
            EffUnixTime_InnerVessel = self.MinUnixTime_CRYTE107
        if unixtime>self.MaxUnixTime_CRYTE107:
            EffUnixTime_InnerVessel = self.MaxUnixTime_CRYTE107
        return self.inter_CRYTE107 (EffUnixTime_InnerVessel)
