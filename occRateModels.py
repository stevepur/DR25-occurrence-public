import numpy as np
import scipy.special as scsp
import sys
sys.path.insert(0, '..')
import occRateUtils as ut

class populationModel:
    def getLabels(self):
        return self.labels

    def getBounds(self):
        return self.bounds
        
    def setBounds(self, bounds):
        self.bounds = bounds

class dualPowerLaw(populationModel):
    def __init__(self):
        self.name = "dualPowerLaw"
        self.bounds = [(0, 50), (-5, 5), (-5, 5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$"]

    def rateModel(self, x, y, xRange, yRange, theta):
        #        f0, alpha, beta = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1))
        return r
            
    def integrate(self, intXLimits, intYLimits, theta, cs):
#        f0, alpha, beta = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1))
        return r
            
    def initRateModel(self):
        f0 = 0.75
        alpha = -0.53218
        beta = -0.5
        theta = [f0, alpha, beta]
        return theta
        
class gaussiansxPowerLaw(populationModel):
    # We fix g1 = 1 to break the degeneracy with F_0
    
    def __init__(self):
        self.name = "gaussiansxPowerLaw"
        #                F_0   alpha     mu_1     sigma_1  g_2    mu_2   sigma_2
        self.bounds = [(0, 5), (-5, 5), (0.5,2), (0, 1), (0, 1), (2, 3), (0, 0.5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\mu_1$", r"$\sigma_1$", r"$g_2$", r"$\mu_2$", r"$\sigma_2$"]

    def rateModel(self, x, y, xRange, yRange, theta):
        f0, alpha, mu1, sig1, g2, mu2, sig2 = theta
        ap1 = alpha+1;
        s2s1 = np.sqrt(2)*sig1
        s2s2 = np.sqrt(2)*sig2
        srpi = np.sqrt(np.pi)
        ymmu1 = y - mu1
        ymmu2 = y - mu2
        
        Cinv = 0.5*((xRange[1]**ap1-xRange[0]**ap1)/ap1) \
            * ( srpi*s2s1*(scsp.erf((yRange[1] - mu1)/s2s1) - scsp.erf((yRange[0] - mu1)/s2s1)) \
              + g2*srpi*s2s2*(scsp.erf((yRange[1] - mu2)/s2s2) - scsp.erf((yRange[0] - mu2)/s2s2)) )
        C = 1./Cinv

        r = f0*C*(x**alpha)*(np.exp(-ymmu1*ymmu1/(s2s1*s2s1)) + g2*np.exp(-ymmu2*ymmu2/(s2s2*s2s2)))
        return r
            
    def integrate(self, intXLimits, intYLimits, theta, cs):
        return ut.integrateRateModel(intXLimits, intYLimits, theta, self, cs)

    def initRateModel(model):
        f0 = 0.75
        alpha = -0.69
        mu1 = 1.5
        sig1 = 0.2
        g2 = 0.5
        mu2 = 2.25
        sig2 = 0.2
        theta = [f0, alpha, mu1, sig1, g2, mu2, sig2]
        return theta

