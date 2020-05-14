#cython: language_level=3
import numpy as np
import datetime


class MyFormPars:

    def __init__(self):
        self.Limits=[]
        self.SetParLimits()
        return

    def GetMinTimeStamp(self):
        return '05/18/16 11:30:00 '
    
    def GetMaxTimeStamp(self,DaysAfterLastPoint=30):
        d=datetime.datetime.now() + datetime.timedelta(DaysAfterLastPoint)
        #return d.strftime('%m/%d/%y %H:%M:%S ')
        return '08/01/18 00:00:00 '
    
    def GetS1ExponentialConstant(self):
        return 2040.6
    
    # function to correct lifetime values found with pax_v6.2.0 and earlier
    # from https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=greene:update_electron_lifetime_model
    def GetLifetimeCorrectionPAX(self):
        CorrectBeforeTime = 1478000000.
        LifetimeCorrection = 6.72734759404e-05
        LifetimeCorrectionUncertainty = 1.56664061785e-06
    
        return (CorrectBeforeTime, LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetOldPAXUnixTime(self):
        CorrectBeforeTime = 1478000000.
        return CorrectBeforeTime
    
    def GetVoltageChangeUnixTime(self):
        CorrectBeforeTime = 1474052400.
        return CorrectBeforeTime
    
    
    # if was calculated before this time, slightly off because of PoRn combined fit
    def GetPoRnCorrection(self):
        CorrectBeforeTime = 1486124188.
        LifetimeCorrection = 2.19733154993e-05
        LifetimeCorrectionUncertainty = 7.67416766802e-06
    
        return (CorrectBeforeTime, LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetLifetimeCorrectionPAX(self):
        CorrectBeforeTime = 1478000000.
        LifetimeCorrection = 6.72734759404e-05
        LifetimeCorrectionUncertainty = 1.56664061785e-06
    
        return (CorrectBeforeTime, LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetSSCorrectionSys(self):
        C = self.GetS1ExponentialConstant()
        LifetimeCorrection = 1./C
        LifetimeCorrectionUncertainty = 0.0009 #temporary, difference between two periods -- 897us and 2040us is treated as systematic error
        return  (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetPAXCorrectionSys(self):
        LifetimeCorrection = 6.72734759404e-05
        LifetimeCorrectionUncertainty = 1.56664061785e-06
        return  (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    # if was calculated before this time, slightly off because of PoRn combined fit
    def GetPoRnCorrectionSys(self):
        LifetimeCorrection = 2.19733154993e-05
        LifetimeCorrectionUncertainty = 7.67416766802e-06
    
        return (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetBeforeSR0FieldCorrectionSys(self):
        LifetimeCorrection =   (0.000128245- ( 0.0002650594016-0.000128245  )*(15.-12)/(12.-8))         #Use Kr correction @2017/2/2 for SR1
        LifetimeCorrectionUncertainty =(0.000128245- ( 0.0002650594016-0.000128245  )*(15.-12)/(12.-8)) #Use Kr correction @2017/2/2 for SR1
        return (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    
    def GetSR0FieldCorrectionSys(self):
        LifetimeCorrection = 0.000128245 
        LifetimeCorrectionUncertainty =0.000128245
        return (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    def GetSR1FieldCorrectionKrSys(self,Unixtimes):
        #Just for test: very rough value to corrected for SR1
        LifetimeCorrectionRn  = 0.0 #Change the treat of 1sigma
        LifetimeCorrectionKr  =  - 0.001267 + 1.031e-12 * Unixtimes
        s1=1.69e-13 #np.sqrt(2.86e-26)   #from finear itting result
        s2=0.000255 #np.sqrt(6.48e-08)
        cov=-4.31e-17  
        deltaKr2 = (Unixtimes*s1)*(Unixtimes*s1) + s2*s2 + 2*Unixtimes*cov
        LifetimeCorrection  =  (LifetimeCorrectionRn + LifetimeCorrectionKr)/2.
        LifetimeCorrectionUncertainty =np.sqrt((np.abs(LifetimeCorrectionRn - LifetimeCorrectionKr)/2.)**2 + 0.00003068**2  + deltaKr2)
    
        return (LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    
    # if was calculated before this time, slightly off because of PoRn combined fit
    def GetPoRnCorrection(self):
        CorrectBeforeTime = 1486124188.
        LifetimeCorrection = 2.19733154993e-05
        LifetimeCorrectionUncertainty = 7.67416766802e-06
    
        return (CorrectBeforeTime, LifetimeCorrection, LifetimeCorrectionUncertainty)
    
    
    def GetKrCorrection(self,ChangeType):
        # below values are generated from correcting Rn222 alphas to Kr for SR1 through mid-August
        if ChangeType == 'LinearToInverse':
            ChangeVal = [-4.770858853409951e-4, 1.1952450931189382]
            ChangeValErr = [4.4695573103954733e-05, 0.02583322436244706]
    
        return (ChangeVal, ChangeValErr)
    
    def GetKrCorrectionSys(self):
        # below values are generated from correcting Rn222 alphas to Kr for SR1 through mid-August
        ChangeVal = [-4.770858853409951e-4, 1.1952450931189382]
        ChangeValErr = [4.4695573103954733e-05, 0.02583322436244706]
    
        return (ChangeVal, ChangeValErr)
    
    
    def GetParametersForHE(self):
        p0 = 100.967
        p1 = -0.118667
        #from three points calculated by Yun
        return p0,p1
    
    def GetCathodeVoltages(self):
        CathodeVoltages = [
                [[0, 1468427280], [15., 15.]],
                [[1468427280, 1469186520], [0., 0.]],
                [[1469186520, 1474043700], [15., 15.]],
                [[1474043700, 1474624740], [13., 13.]],
                [[1474624740, 1484731512], [12., 12.]],
                [[1484731512, 1484942279], [0., 0.]],
                [[1484942279, 1485445141], [9., 10.]],
                [[1485445141, 1485802500], [0., 0.]],
                [[1485802500, 1486054320], [7., 7.]],
                [[1486054320, 1496653500], [8., 8.]],
                [[1496653500, 1496661420], [0., 0.]],
                [[1496661420, 1519504980], [8., 8.]],
                [[1519504980, 1528209780], [1., 10.]],
                [[1528209780, 1528528020], [0., 0.]],
                [[1528528020, 1528784640], [8., 8.]],
                [[1528784640, 1528966320], [0., 0.]],
                [[1528966320, int(2**32-1)], [1., 10.]]
                ]
    
        for i in range(len(CathodeVoltages)):
            if i == 0:
                continue
            assert CathodeVoltages[i][0][0] == CathodeVoltages[i-1][0][1]
    
        return CathodeVoltages
    
    def GetSpecialPeriods(self):
        SpecialPeriods = [
                            [1465913920, 1466517600]
                            ]
        return SpecialPeriods
    
    def GetGasOnlyPeriods(self):
        GasOnlyPeriods = [
                            [1471900000, 1472840000],
                            ]
        return GasOnlyPeriods
    
    def GetLowFlowPeriods(self):
        Periods = [
                            [1527591600,1528824600],
                            ]
        return Periods
    
    
    def GetMaximumHeatingPower(self):
        return 260.
    
    def GetHeatingTrends(self):
        trends = np.asarray([
            [[0, 1476210000], [0, 0]],
            [[1476210000, 1496520000], [-5.43424734e-07, 8.88309454e+01]],
            [[1496520000, 1503241000], [-2.00236315e-06, 7.80391177e+01]],
            [[1503241000, 1511180000], [-2.00236315e-06, 7.40391177e+01]],
            [[1511180000, 1523870000], [-2.64408198e-07, 5.94658620e+01]],
            [[1523870000, 1528910000], [-2.70867857e-06, 4.50975665e+01]],
            [[1528910000, 2000000000], [-2.30794789e-06, 2.69333682e+01]]
        ])
        for i in range(len(trends)):
            if i == 0:
                continue
            if trends[i][0][0] != trends[i-1][0][1]:
                raise ValueError('Heating trends are not continuous')
        return trends
    
    
    def GetPurityDrops(self):
        PurityDrops = dict(
                        unixtimes = [
                                    1465937520,   # the amount impurity changed during power event
                                    1468597800,   # the amount impurity changed during LN2 test @ July 15, 2016
                                    1477544880,   # Kr distillation start at 10/27 2016
                                    1479772379,   # amount impurity changed during power glitch @ Nov. 21, 2016`
                                    1485951100,   # the amount impurity after earthquake in late January 2017 (Feb.3, from level change)

                                    1496685600,   # amount of impurity from gate washing on June 5, 2017
                                    1519205760,   # from power glitch on February 21, 2018, Rn220 calib
                                    1519515240,   # from blackout on February 24, 2018
                                    1520444280,   # from gate washings on March 7, 2018
                                    1531821600,   # from Rn220 calibration July 17 - 23, 2018

                                    1482175745,   # from Rn220 calibration in 2016, 12-19 to 12-21 
                                    1508942012,   # from Rn220 calibration October 2017 Rn-220 calibration 
                                    1481570400,   # Kr Distillation end
                                    1476360000,   # from Rn220 calibration 13/10 - 17/10, 2016
                                    1472050000,   # QDrive Update 2016 

                                    1470485940,   # From Liquid level change 8/6 12:19 
                                    1475180000    # PurPipe upgrade on 2016

                            ],
                            
                        #0: fit both as free
                        #1: fit only for LX2,
                        #2: fit with barance at heat exchanger,
                        #3: fit only for GXe
                        types = [
                        2,
                        1,
                        1,
                        2,
                        1,

                        1,
                        2,
                        2,
                        1,
                        2,

                        2,
                        2,
                        1,
                        2,
                        1,

                        1,
                        1,
                        ],
                        valuesLXe = [
                        10.0613537,
                        3.36014333,
                        4.0e-1, 
                        1.0e-1,
                        1.0e-1,
                        1.e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1.0e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1e-1],
    
                        valuesGXe = [10.0613537, 3.36014333,
                        4.0e-1, 
                        1.0e-1,
                        1.0e-1,
                        1.e-1,
						1e-1,
						1e-1,
						1e-1,
                        1.0e-1,
						1e-1,
						1e-1,
						1e-1,
                        1e-1,
                        1e-1,
                        1e-1,
                        1e-1
                        ]
    
                        )
        assert len(PurityDrops['unixtimes']) == len(PurityDrops['valuesLXe'])
        assert len(PurityDrops['unixtimes']) == len(PurityDrops['valuesGXe'])
    
        return PurityDrops
    
    
    def GetScienceRunUnixtimes(self):
        ScienceRunUnixtimes = dict(
                                SR0 = [1477078362, 1484731512], # include SR0 AmBe calibration
                                SR1 = [1486054320, 1517964780]
                                )
        return ScienceRunUnixtimes
    
    
    def GetOperations(self,XLimStart, ShowLegend=True, PlotRnEvolution=True, ShowResiduals=True):
        Operations = {
            'PowerOutage': {
                'unixtime': 1465937520,
                'y': 'top',
                'print_text_after': 2,
                'text': 'PTR warm-up',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'draw': True,
            },
            'LNTest': {
                'unixtime': 1468597800,
                'y': 'top',
                'print_text_after': 2,
                'text': 'LN$_2$ Cooling Test',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'draw': True,
            },
            'GXeCirculation': {
                'unixtime': 1471880000,
                'y': 'top',
                'print_text_after': 2,
                'text': 'GXe-Only Circulation',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'draw': True,
            },
            'PURUpgrade': {
                'unixtime': 1475180000,
                'y': 'top',
                'print_text_after': 2,
                'text': 'PUR upgrade',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'draw': True,
            },
            'PowerGlitch1': {
                'unixtime': 1479772379,
                'y': 'top',
                'print_text_after': 2,
                'text': 'Power Disruption',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'top',
                'linewidth': 2,
                'draw': True,
            },
            'Rn220Injection': {
                'unixtime': 1482143819,
                'y': 'top',
                'print_text_after': 2,
                'text': '$\\rm{^{220}Rn\,\, Injection}$',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'top',
                'linewidth': 2,
                'draw': True,
            },
            'Earthquake': {
                'unixtime': 1484731512,
                'y': 'bottom',
                'print_text_after': 1,
                'text': 'Earthquake',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'GateWashing1': {
                'unixtime': 1496685600,
                'y': 'bottom',
                'print_text_after': 2,
                'text': 'Detector Maintenance',
                'size': 22,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
            },
            'PowerGlitch2': {
                'unixtime': 1519205760,
                'y': 'bottom',
                'print_text_after': -4,
                'text': 'Power Disruption',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': False,
            },
            'Blackout': {
                'unixtime': 1519515240,
                'y': 'bottom',
                'print_text_after': 0.5,
                'text': 'Blackout',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': False,
            },
            'GateWashing2': {
                'unixtime': 1520263920,
                'y': 'bottom',
                'print_text_after': 2,
                'text': '',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'GateWashing3': {
                'unixtime': 1520349480,
                'y': 'bottom',
                'print_text_after': 2,
                'text': '',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'GateWashing4': {
                'unixtime': 1520444280,
                'y': '480',
                'print_text_after': 1,
                'text': 'Gate Washings',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'CirculationPipesUpgrade': {
                'unixtime': 1523318400,
                'y': '480',
                'print_text_after': 1,
                'text': 'Purification Pipes Upgrade',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'CirculationStopped': {
                'unixtime': 1527591600,
                'y': '480',
                'print_text_after': 1,
                'text': 'Circulation Stopped',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'CirculationRestarted': {
                'unixtime': 1528221600,
                'y': '480',
                'print_text_after': 1,
                'text': 'Circulation Restarted (2 QDrives)',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
            'CirculationMag': {
                'unixtime': 1528813800,
                'y': '480',
                'print_text_after': 1,
                'text': 'Magnetic Pump Installation',
                'size': 32,
                'rotation': 'vertical',
                'color': 'grey',
                'vertical_alignment': 'bottom',
                'linewidth': 2,
                'draw': True,
            },
        }
        if ShowLegend:
            if PlotRnEvolution:
                if XLimStart == 'SR1':
                    Operations['GateWashing1']['print_text_after'] = -6
                elif XLimStart == 'sr0_ambe':
                    Operations['GateWashing1']['print_text_after'] = -17
                    Operations['GateWashing1']['text'] =  'Detector\nMaintenance'
                    Operations['GateWashing1']['y'] = 451
            if not ShowResiduals and XLimStart == 'sr0_ambe':
                Operations['GateWashing1']['y'] = 418
                Operations['GateWashing1']['print_text_after'] = -7
    
        return Operations
    
    # parameter selection
    # select which parameters to fit
    def GetDefaultPars(self):
        PurityDrops = self.GetPurityDrops()
        default_pars = [
            3.7e-3, #0, attaching rate from literature
            5.09257628e+01,  #1, initial LXe concentration
            4.02212420e-01,  #2, impurity attaching prob for vaporization
            4.10390827e-01,  #3, impurity attaching prob for condensation
            [
            1.90294571e+02, #4, GXe volume outgassing, in unit of kg*ppb/day
            ],
            [
            1.94252374e+02, #5, LXe volume outgassing, in unit of kg*ppb/day
                1./1000.,          # LXe outgassing exponential term, in days
            1.94252374e+02,  # LXe volume outgassing, in unit of kg*ppb/day
                1./1000.,          # LXe outgassing Linear term, in days
            ],
            PurityDrops['unixtimes'], #6, time for the impurity change, after correction
            PurityDrops['valuesGXe'], #7
            PurityDrops['valuesLXe'], #8
            PurityDrops['types'], #9
            [1480344349 - 148680.0 - 0.15*86400,1480810387 + 2.4*86400, 0.98], #10 periods when getter is suspected to have lowered efficiency, roughly from 11-26 to 12-05
            [0.975,0.5], #11 Flow Fraction of outgassing
            30, #12 Volatility parameter
            [0, 0, 0, 0, 0, 0, 0], #13 systematics for common data corrections
            ]
    
        return default_pars
    
    
    def FormPars(self,x):
        #print(self.Limits)
        pars = self.GetDefaultPars()
        PurityDrops = self.GetPurityDrops()
        cdef:
            int NumGXeOutgassingTerms = len(pars[4])
            int NumLXeOutgassingTerms = len(pars[5])
            int NumOutgassingTerms = NumGXeOutgassingTerms + NumLXeOutgassingTerms

            int NumPurityDrops = len(PurityDrops['unixtimes'])
            int pindex = 3+NumOutgassingTerms
            int zeroindex = PurityDrops['types'].count(0)

            int NumTerms = NumPurityDrops +zeroindex+ NumOutgassingTerms
        
        #if len(x)<13+NumPurityDrops:
        #    return pars
        IfOutOfBoundary = False 
        OutOfBoundaryParIndex = []
        for i in range(len(x)):
            lo = self.Limits[i][0]
            hi = self.Limits[i][1]
            y=x[i]
            if lo is not None:
                if y<lo:
                    IfOutOfBoundary = True
                    OutOfBoundaryParIndex.append(i)
            if hi is not None:
                if y>hi:
                    IfOutOfBoundary = True
                    OutOfBoundaryParIndex.append(i)
        pars[1] = x[0] # initial LXe concentration
        pars[2] = x[1] # vaporization attaching prob
        pars[3] = x[2] # condensation attaching prob
        for i in range(NumGXeOutgassingTerms):
            pars[4][i] = x[3+i]
        for i in range(NumLXeOutgassingTerms):
            pars[5][i] = x[3+i+NumGXeOutgassingTerms]
        for i in range(NumPurityDrops):
            pars[7][i] = x[pindex+i]*1e-5
            pars[8][i] = x[pindex+i]*1e-5

        pars[10][2]  = x[3+NumTerms] # lowered efficiency in Nov 2016 (no clear reason) 
        pars[11][0]  = x[4+NumTerms] # Flow fraction gas
        pars[11][1]  = x[5+NumTerms] # Flow fraction liquid 
        pars[12]    = x[6+NumTerms] # Volatility Parameter
        pars[-1][0] = x[-7]
        pars[-1][1] = x[-6]
        pars[-1][2] = x[-5]
        pars[-1][3] = x[-4]
        pars[-1][4] = x[-3]
        pars[-1][5] = x[-2]
        pars[-1][6] = x[-1]
        return (pars, IfOutOfBoundary,OutOfBoundaryParIndex)
    
    def GetInitialBounds(self,x0,x0_step):
        lo_bound = x0-2*x0_step
        hi_bound = x0+2*x0_step
        for i in range(len(x0)):
            lo = self.Limits[i][0]
            hi = self.Limits[i][1]
            if lo is not None:
                lo_bound[i] = np.fmax(lo_bound[i],lo)
            if hi is not None:
                hi_bound[i] = np.fmin(hi_bound[i],hi)
        return (lo_bound, hi_bound)
    
    def SetParLimits(self):
        pre_Limits = []
        pars = self.GetDefaultPars()
        PurityDrops = self.GetPurityDrops()
        x,xstep = self.GetOldBestParameters()
        cdef:
            int NumGXeOutgassingTerms = len(pars[4])
            int NumLXeOutgassingTerms = len(pars[5])
            int NumOutgassingTerms = NumGXeOutgassingTerms + NumLXeOutgassingTerms
            int NumPurityDrops = len(PurityDrops['unixtimes'])
            int pindex = 3+NumOutgassingTerms
            int zeroindex = PurityDrops['types'].count(0)
            int NumTerms = NumPurityDrops +zeroindex+ NumOutgassingTerms
            int NumSys = len(pars[-1])
        #print(NumTerms)
        for i in range(0,len(x)):
            lo_limits=None
            hi_limits=None
            if not (i==3 or i>=(len(x)-NumSys)): #Outgass change, common sys params, date change
                lo_limits=0
            if i==6+NumTerms:
                lo_limits=1.0
            if i==3:
                lo_limits=-4*0.11
                hi_limits=4*0.11
            if i==5:
                lo_limits=1./200
            if i==7:
                hi_limits=1./200
            if i==1:
                lo_limits=0.1
            if (i==1 or i ==2 or i==3+NumTerms):
                hi_limits=1.0
    
            pre_Limits.append((lo_limits,hi_limits))

        self.Limits = pre_Limits
    
        return
    
    def GetParLimits(self):
        return self.Limits
    
    def GetOldBestParameters(self):
        x0 = np.array([
             11.4     , # 0 initial LXe concentration                              
             0.3      , # 1 impuritiy attachment prob for vaporization             
             0.05      , # 2 impuritiy attachment prob for condensation             
             0.1      , # 3 GXe volume outgassig, in units of kg/day               
             1500     , # 4 LXe volume outgassig, in units of kg/day              
    
             1./100    , # 5 LXe outgassing exponential term, in days              
             1500     , # 6 LXe volume outgassig, in units of kg/day              
             1./10000  , # 7 LXe outgassing exponential term, in days              
             9.58     , # 8 2,LXE: impurity change from power event  
             2.92     , # 9 1,LXE: impurity change from LN2 test July 15, 2016                                         

             0.5      , # 10 1,LXE: Kr distillation start
             0.282    , # 11 2,LXE: impurity change from power glitch Nov. 21, 2016                                     
             0.103    , # 12 1,LXE: 2nd impurity change after earthquake late January, 201                               
             0.109    , # 13 1,LXE: impurity change from gate washing June 5, 2017                                           
             0.431    , # 14 2,LXE: Rn220 calibration and impurity change from power glitch February 21, 2018

             0.124    , # 15 2,LXE: impurity change from blackout February 24, 2018                                          
             0.21     , # 16 1,LXE: impurity change from gate washing March 7, 2018                                          
             0.21     , # 17 2,LXE: from Rn220 calibration July 17 - 23, 2018
             0.21     , # 18 2,LXE: # from Rn220 calibration in 2016, 12-19 to 12-21 
             0.21     , # 19 2,LXE: # from Rn220 calibration October 2017 Rn-220 calibration 

             0.21     , # 20 1,LXE: Kr distillation end 
             0.21     , # 21 2,LXE: # from Rn220 calibration 13/10 - 17/10, 2016
             0.21     , # 22 1,LXE: # QDrive Update 2016 
             0.21     , # 23 1,LXE: # PurPipe upgrade
             0.21     , # 24 1,LXE: # from Level change 
    
             0.865    , # 25 lowered efficiency Nov. 28 to Dec. 6                  
             0.8      , # 26 Flow Fraction Gas
             0.8      , # 27 Flow Fraction Liquid
             30       , # 28 Volatility parameter
             0        , # 29 common uncertainty                                   
    
             0        , # 30 common uncertainty                                   
             0        , # 31 common uncertainty                                   
             0        , # 32 common uncertainty                                   
             0        , # 33 common uncertainty                                   
             0        , # 34 common uncertainty                                   

             0        , # 35 common uncertainty                                   
    		])
        x0_steps = np.array([
            5      , # 1 initial LXe concentration
            0.05       , # 2 impuritiy attachment prob for vaporization
            0.01       , # 3 impuritiy attachment prob for condensation
            1000.     , # 4 GXe volume outgassing, in units of kg/day
            1000      , # 10 LXe volume outgassig, in units of kg/day
                           
            1./100    , # 11 LXe outgassing exponential term, in days
            1000      , # 12 LXe volume outgassig, in units of kg/day
            1./1000    , # 13 LXe outgassing exponential term, in days
            9.58     , # 14 3,GXE: impurity change from power event  
            2.92     , # 15 3,GXE: impurity change from LN2 test July 15, 2016                                         

            0.5      , # 16 0,LXE: Kr distillation start
            0.282    , # 17 3,GXE: impurity change from power glitch Nov. 21, 2016                                     
            0.103    , # 19 0,LXE: 2nd impurity change after earthquake late January, 201                               
            0.109    , # 20 1,LXE: impurity change from gate washing June 5, 2017                                           
            0.431    , # 21 2,LXE: Rn220 calibration and impurity change from power glitch February 21, 2018

            0.124    , # 22 0,LXE: impurity change from blackout February 24, 2018                                          
            0.21     , # 23 1,LXE: impurity change from gate washing March 7, 2018                                          
            0.21     , # 24 2,LXE: from Rn220 calibration July 17 - 23, 2018
            0.21     , # 25 2,LXE: # from Rn220 calibration in 2016, 12-19 to 12-21 
            0.21     , # 26 2,LXE: # from Rn220 calibration October 2017 Rn-220 calibration 

            0.21     , # 28 0,LXE: Kr distillation end 
            0.21     , # 29 2,LXE: # from Rn220 calibration 13/10 - 17/10, 2016
            0.21     , # 28 0,LXE: # QDrive Update 2016 
            0.21     , # 28 0,LXE: # PurPipe upgrade
            0.21     , # 28 0,LXE: # Level change

            0.5       , # 31 lowered efficiency Nov. 28 to Dec. 6
            0.2       , # 35 Flow Fraction Gas
            0.2       , # 36 Flow Fraction Liquid
            400       , # 51 Volatility parameter
            1         , # 40  
                            
            1         , # 40  
            1         , # 41  
            1         , # 42  
            1         , # 43  
            1         , # 45  

            1         , # 46  
            ])
    
        return (x0, x0_steps)
    
    def GetParDescriptions(self):
        x0 = np.array([
                "initial LXe concentration"
                ,"relative cryo power used to condensation"
                ,"impuritiy attachment prob for condensation"
                ,"GXe volume outgassig, in units of kg/day"
                ,"LXe volume outgassig, in units of kg/day"

                ,"LXe outgassing exponential term, in 1/days"
                ,"LXe volume outgassig, in units of kg/day"
                ,"LXe outgassing exponential term, in 1/days"
                ,"the amount impurity changed during power event " 
                ,"the amount impurity changed during LN2 test @ July 15, 2016 " 

                ,"Kr distillation start at 10/27 " 
                ,"amount impurity changed during power glitch @ Nov. 21, 2016` " 
                ,"the amount impurity after earthquake in late January 2017 (Feb.3, from level change) " 
                ,"amount of impurity from gate washing on June 5, 2017 " 
                ,"From power glitch on February 21, 2018 and Rn220" 

                ,"From blackout on February 24, 2018 " 
                ,"From gate washings on March 7, 2018 " 
                ,"From Rn220 calibration July 17 - 23, 2018 " 
                ,"From Rn220 calibration in 2016, 12-19 to 12-21 " 
                ,"From Rn220 calibration October 2017 Rn-220 calibration " 

                ,"Kr Distillation end " 
                ,"From Rn220 calibration 13/10 - 17/10, 2016 " 
                ,"Qrive Update" 
                ,"From Liquid Level Change" 
                ,"PURPipe Update (2018)" 
    
                ,"lowered efficiency Nov. 28 to Dec. 6 2016"
                ,"Flow coeffecient of GXe circulation vs outgassing source"
                ,"Flow coefficient of LXe circulation vs outgassing source"
                ,"Volatility parameter"
                ,"common uncertainty SS-WT"

                ,"common uncertainty SS"
                ,"common uncertainty PAX"
                ,"common uncertainty PoRn"
                ,"common uncertainty EF Before SR0"
                ,"common uncertainty EF SR0"

                ,"common uncertainty EF SR1"
    		])
        return x0
    
   
    def GetImpactfulUnixtimes(self):
            ScienceRunUnixtimes = self.GetScienceRunUnixtimes()
            CathodeVoltages = self.GetCathodeVoltages()
            default_pars = self.GetDefaultPars()
            PurityDrops = self.GetPurityDrops()
            UnixTimePDs = PurityDrops['unixtimes']
    
            ImpactfulUnixtimes = []
            for ScienceRunUnixtimes in ScienceRunUnixtimes.values():
                for ScienceRunUnixtime in ScienceRunUnixtimes:
                    if ScienceRunUnixtime > 2e9:
                        continue
                    ImpactfulUnixtimes.append(ScienceRunUnixtime)
    
            for CathodeVoltage in CathodeVoltages:
                if CathodeVoltage[0][0] != 0:
                    ImpactfulUnixtimes.append(CathodeVoltage[0][0])
    
            for UnixTimePD in UnixTimePDs:
                ImpactfulUnixtimes.append(UnixTimePD)
    
            ImpactfulUnixtimes.append(default_pars[11][0])
            ImpactfulUnixtimes.append(default_pars[11][1])
    
            return ImpactfulUnixtimes
