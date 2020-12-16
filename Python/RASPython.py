import numpy as np
import scipy as sc
from math import sqrt

# No inverse update in this implementation

def RASPython(F,c,epsilon):
    
    N = F.shape[1]
    nbFF = F.shape[0]

    idS = np.zeros(N,int)
    weights = np.zeros(N,float)

    R = np.empty((nbFF,nbFF),float)

    lev_score = np.sum(np.multiply(F,F),axis = 0)

    nbSamples = 0;
    t = 0.
    print('iterations ...')
    print('Inverse Update = False')

    for i in range(N-1):

        if i%1000 == 0:
            print(nbSamples)


        if nbSamples == 0:
            p_i = min(c*(1/epsilon)*(1+t)*lev_score[i],1)

            if np.random.rand(1)[0] < p_i:
                nbSamples = 1
                idS[nbSamples-1] = i
                weights[nbSamples-1] = 1/sqrt(p_i)
                R[:nbSamples,:nbSamples] = 1/( lev_score[i] +epsilon/(weights[nbSamples-1])**2 )
        else:
            
            temp = np.transpose(F[:,[i]])
            PS_i = np.dot(temp,F[:,idS[:nbSamples]])
            
            R_temp = R[:nbSamples,:nbSamples];
            T = np.sum(np.dot(PS_i,R_temp)*PS_i,1)
                
            p_i = min(1, c*(1/epsilon)*(1+t)*max(lev_score[i]-T,0))

            if np.random.rand(1)[0] < p_i:
                nbSamples = nbSamples + 1
                idS[nbSamples-1] = i
                weights[nbSamples-1] = 1/sqrt(p_i)

                SPS = np.dot(np.transpose(F[:,idS[:nbSamples]]),F[:,idS[:nbSamples]])
                SPS_reg = SPS + np.diag(np.divide(epsilon,np.square(weights[:nbSamples])))

                R[:nbSamples,:nbSamples] = sc.linalg.solve(SPS_reg, np.eye(nbSamples),True)
             
             
    return idS[:nbSamples]

