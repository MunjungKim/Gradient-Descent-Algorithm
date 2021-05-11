import numpy as np
import math


def dowlist(ts,p):
    """
    
    
    
    """
    size = ts.shape[0] # sample size

    X = np.zeros((1,p))
    Y = np.zeros(shape=(1,))
    x = np.zeros(shape=(1,p))
    y = np.zeros(shape=(1,))


    for i in range(size-p-1):
        if math.isnan(ts[i]): #when the value is nan
                continue
        else: # when the value is not nan
            for m in range(p):
                x[0,m]=ts[i+m]
            if np.sum(np.isnan(x))>0 :
                continue
            y=ts[i+p] # 다음달 co2 concentration
            if math.isnan(y): # y값이 NaN이면 무시하고 다시 새로 시작
                continue
            X=np.vstack([X,x])
            Y=np.vstack([Y,y])
            
    return X[1:,:],Y[1:,:]

def dowlist_10(ts,p):
    """
    
    
    
    """
    size = ts.shape[0] # sample size

    X = np.zeros((1,p))
    Y = np.zeros(shape=(1,))
    x = np.zeros(shape=(1,p))
    y = np.zeros(shape=(1,))


    for i in range(size-p-1-10):
        if math.isnan(ts[i]): #when the value is nan
                continue
        else: # when the value is not nan
            for m in range(p):
                x[0,m]=ts[i+m]
            if np.sum(np.isnan(x))>0 :
                continue
            y=ts[i+p+10] # 다음달 co2 concentration
            if math.isnan(y): # y값이 NaN이면 무시하고 다시 새로 시작
                continue
            X=np.vstack([X,x])
            Y=np.vstack([Y,y])
            
    return X[1:,:],Y[1:,:]