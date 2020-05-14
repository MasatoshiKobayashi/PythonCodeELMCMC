import Fit_func2
import numpy as np

Fit_func = Fit_func2.MyFit_func()

def SetFunc(func):
    print("Setting functions...")
    global Fit_func
    Fit_func= func
    return 


def LnLike(x):
    LnL = Fit_func.LnLikeDataCorrections(x)
    #print("LnL:",LnL)
    return float(LnL)

index=0
def dumpTXT(output,UnixTimes, Taus,P=False):
    fout = open(output, 'w')
    if P:
        for par in Taus:
            fout.write("{0}\n".format(par))
        fout.close()
    else:
        for unixtime, tau, lower, upper in zip(UnixTimes, Taus, Taus, Taus ):
            fout.write("{0}\t\t{1}\t\t{2}\t\t{3}".format(unixtime,tau,lower,upper))
            fout.write("\n")
        fout.close()


def LnLikeChi2(x):
    global index
    LnL = Fit_func.LnLikeDataCorrections(x)
    if(index%10==0):
        print(index, -2*LnL)
        if(index%300==0):
            print(np.asarray(x))
            #NumOfInterpolation = 1000
            #UnixTimes = np.linspace(MinUnixTime, MaxUnixTime, NumOfInterpolation)
            #Trends,Ig,Il,Og,Ol,Vapo,Cond,FlowCond=Fit_func.GetTrends(UnixTimes,x)
            #fout = FitOutputTXT+".txt"
            #dumpTXT(fout,UnixTimes,Trends)
            #fout = FitOutputTXT+"_Ig.txt"
            #dumpTXT(fout,UnixTimes,Ig)
            #fout = FitOutputTXT+"_Il.txt"
            #dumpTXT(fout,UnixTimes,Il)
            fout ="TEMP_Param.txt"
            dumpTXT(fout,x,x,P=True)
    index+=1
    return -2*LnL
