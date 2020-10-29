import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display


class populationModel:
    def getLabels(self):
        return self.labels

    def getBounds(self):
        return self.bounds
        
    def setBounds(self, bounds):
        self.bounds = bounds

class dualPowerLaw(populationModel):
    def __init__(self, cs = None):
        self.name = "dualPowerLaw"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$"]

    def rateModel(self, x, y, z, xRange, yRange, zRange, theta):
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
                        
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
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
            
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
        return self.rateModel(x, y, 0, cs.periodRange, cs.rpRange, cs.tempRange, theta)
            
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
        return self.integrate(intXLimits, intYLimits, [0, 1], theta, cs)


    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        theta = [f0, alpha, beta]
        return theta
        

class dualPowerLawFixedTeff(populationModel):
    def __init__(self, cs = None):
        self.name = "dualPowerLawFixedTeff"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$"]
        self.teffBreak = 5117
        self.teffExp = [3.16, 4.49]
        self.teffNorm = [10**(-11.839), 10**(-16.769)]

    def rateModel(self, x, y, teff, xRange, yRange, tRange, theta):
        f0, alpha, beta = theta
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1
        teffNorm = self.teffNorm[0]*(self.teffBreak**ta0p1 - tRange[0]**ta0p1)/ta0p1 \
                    +self.teffNorm[1]*(tRange[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        
        ap1 = alpha+1;
        bp1 = beta+1;
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1)) \
            * teffFactor/teffNorm
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1
        teffNorm = self.teffNorm[0]*(self.teffBreak**ta0p1 - cs.tempRange[0]**ta0p1)/ta0p1 \
                    +self.teffNorm[1]*(cs.tempRange[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *(self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
            +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1)/teffNorm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1
        teffNorm = self.teffNorm[0]*(self.teffBreak**ta0p1 - cs.tempRange[0]**ta0p1)/ta0p1 \
                    +self.teffNorm[1]*(cs.tempRange[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1

        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *(self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
            +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1)/teffNorm
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1
        teffNorm = self.teffNorm[0]*(self.teffBreak**ta0p1 - cs.tempRange[0]**ta0p1)/ta0p1 \
                    +self.teffNorm[1]*(cs.tempRange[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor/teffNorm
        return r
            
    def integrateX(self, intXLimits, y, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        teffFactor = np.zeros(teff.shape)
        teffFactor[teff <= self.teffBreak] = (teff[teff <= self.teffBreak]/5778)**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = (teff[teff > self.teffBreak]/5778)**self.teffExp[1]
        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor
        return r
            
    def integrateY(self, x, intYLimits, z, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        teffFactor = np.zeros(teff.shape)
        teffFactor[teff <= self.teffBreak] = (teff[teff <= self.teffBreak]/5778)**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = (teff[teff > self.teffBreak]/5778)**self.teffExp[1]
        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor
        return r
                        

    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        theta = [f0, alpha, beta]
        return theta
    
class dualPowerLawFixedTeff0(populationModel):
    def __init__(self, cs = None):
        self.name = "dualPowerLawFixedTeff0"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$"]
        self.teffBreak = 5117
        self.teffExp = [3.16, 4.49]
        self.teffNorm = [10**(-11.839), 10**(-16.769)]
        self.teffRef = 5778 # use the Sun for the normalization temperature
        
        if self.teffRef <= self.teffBreak:
            self.norm = self.teffNorm[0]*self.teffRef**self.teffExp[0]
        else:
            self.norm = self.teffNorm[1]*self.teffRef**self.teffExp[1]

    def teffAverage(self, intZLimits):
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1

        if intZLimits[1] < self.teffBreak:
            teffInt = self.teffNorm[0]*(intZLimits[1]**ta0p1 - intZLimits[0]**ta0p1)/ta0p1
        elif intZLimits[0] > self.teffBreak:
            teffInt = self.teffNorm[1]*(intZLimits[1]**ta1p1 - intZLimits[0]**ta1p1)/ta1p1
        else:
            teffInt = self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
                        +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        return  teffInt/(intZLimits[1] - intZLimits[0])

    def getTeffFactor(self, teff):
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        return teffFactor

    def rateModel(self, x, y, teff, xRange, yRange, tRange, theta):
        f0, alpha, beta = theta
        
        teffFactor = self.getTeffFactor(teff)
        
        ap1 = alpha+1;
        bp1 = beta+1;
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1)) \
            * teffFactor/self.norm
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        teffAvg = self.teffAverage(intZLimits)

        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/self.norm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        teffAvg = self.teffAverage(intZLimits)

        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/self.norm
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        teff = np.atleast_2d(teff)
        
        teffFactor = self.getTeffFactor(teff)

        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor/self.norm
        return r

    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        theta = [f0, alpha, beta]
        return theta
        
class dualPowerLawFixedTeffAvg(populationModel):
    def __init__(self, cs = None):
        self.name = "dualPowerLawFixedTeffAvg"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$"]
        self.teffBreak = 5117
        self.teffExp = [3.16, 4.49]
        self.teffNorm = [10**(-11.839), 10**(-16.769)]
        self.teffRef = 5778 # use the Sun for the normalization temperature
        
        self.norm = self.teffAverage(cs.tempRange)

    def teffAverage(self, intZLimits):
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1

        if intZLimits[1] < self.teffBreak:
            teffInt = self.teffNorm[0]*(intZLimits[1]**ta0p1 - intZLimits[0]**ta0p1)/ta0p1
        elif intZLimits[0] > self.teffBreak:
            teffInt = self.teffNorm[1]*(intZLimits[1]**ta1p1 - intZLimits[0]**ta1p1)/ta1p1
        else:
            teffInt = self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
                        +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        return  teffInt/(intZLimits[1] - intZLimits[0])

    def getTeffFactor(self, teff):
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        return teffFactor

    def rateModel(self, x, y, teff, xRange, yRange, tRange, theta):
        f0, alpha, beta = theta
        
        teffFactor = self.getTeffFactor(teff)

        ap1 = alpha+1;
        bp1 = beta+1;
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1)) \
            * teffFactor/self.norm
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        teffAvg = self.teffAverage(intZLimits)

        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/self.norm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        ap1 = alpha+1;
        bp1 = beta+1;
        teffAvg = self.teffAverage(intZLimits)

        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/self.norm
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
        teff = np.atleast_2d(teff)
        
        teffFactor = self.getTeffFactor(teff)

        ap1 = alpha+1;
        bp1 = beta+1;
        
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor/self.norm
        return r

    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        theta = [f0, alpha, beta]
        return theta
         
# same as dualPowerLawFixedTeffAvg but we don't fix the Teff dependence, and fit it instead
class triplePowerLaw(populationModel):
    def __init__(self, cs = None):
        self.name = "triplePowerLaw"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5), (-500, 50)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$", r"$\gamma$"]

    def rateModel(self, x, y, z, xRange, yRange, zRange, theta):
        f0, alpha, beta, gamma = theta

        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;
        teffAvg = (zRange[1]**gp1 - zRange[0]**gp1)/gp1/(zRange[1] - zRange[0])
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1)) \
            * (z**gamma)/teffAvg
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;
        teffNorm = (cs.tempRange[1]**gp1 - cs.tempRange[0]**gp1)/gp1/(cs.tempRange[1] - cs.tempRange[0])
        teffAvg = (intZLimits[1]**gp1 - intZLimits[0]**gp1)/gp1/(intZLimits[1] - intZLimits[0])

        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/teffNorm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;

        teffNorm = (cs.tempRange[1]**gp1 - cs.tempRange[0]**gp1)/gp1/(cs.tempRange[1] - cs.tempRange[0])
        teffAvg = (intZLimits[1]**gp1 - intZLimits[0]**gp1)/gp1/(intZLimits[1] - intZLimits[0])

        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/teffNorm
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;

        teffNorm = (cs.tempRange[1]**gp1 - cs.tempRange[0]**gp1)/gp1/(cs.tempRange[1] - cs.tempRange[0])
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * (teff**gamma)/teffNorm
        return r

    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        gamma = 0
        theta = [f0, alpha, beta, gamma]
        return theta


# combine dualPowerLawFixedTeffAvg and triplePowerLaw
class triplePowerLawTeffAvg(populationModel):
    def __init__(self, cs = None):
        self.name = "triplePowerLawTeffAvg"
        self.bounds = [(0, 50000), (-5, 5), (-5, 5), (-500, 50)]
        self.labels = [r"$F_0$", r"$\beta$", r"$\alpha$", r"$\gamma$"]
        self.teffBreak = 5117
        self.teffExp = [3.16, 4.49]
        self.teffNorm = [10**(-11.839), 10**(-16.769)]
        self.teffRef = 5778 # use the Sun for the normalization temperature
        
    def teffAverage(self, intZLimits, gamma):
        ta0p1 = self.teffExp[0] + gamma + 1
        ta1p1 = self.teffExp[1] + gamma + 1

        if intZLimits[1] < self.teffBreak:
            teffInt = self.teffNorm[0]*(intZLimits[1]**ta0p1 - intZLimits[0]**ta0p1)/ta0p1
        elif intZLimits[0] > self.teffBreak:
            teffInt = self.teffNorm[1]*(intZLimits[1]**ta1p1 - intZLimits[0]**ta1p1)/ta1p1
        else:
            teffInt = self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
                        +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        return  teffInt/(intZLimits[1] - intZLimits[0])

    def getTeffFactor(self, teff):
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        return teffFactor
        
    def rateModel(self, x, y, z, xRange, yRange, zRange, theta):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]

        teffFactor = self.getTeffFactor(z)
        teffNorm = self.teffAverage(zRange, gamma)

        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;
        teffAvg = (zRange[1]**gp1 - zRange[0]**gp1)/gp1/(zRange[1] - zRange[0])
        r = f0*(ap1*(x**alpha)/(xRange[1]**ap1-xRange[0]**ap1)) \
            *(bp1*(y**beta)/(yRange[1]**bp1-yRange[0]**bp1)) \
            * (z**gamma) * teffFactor/teffNorm
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;
        teffNorm = self.teffAverage(cs.tempRange, gamma)
        teffAvg = self.teffAverage(intZLimits, gamma)

        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/teffNorm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;

        teffNorm = self.teffAverage(cs.tempRange, gamma)
        teffAvg = self.teffAverage(intZLimits, gamma)

        r = f0*(ap1*(x**alpha)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *(bp1*(y**beta)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            *teffAvg/teffNorm
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0, alpha, beta, gamma = theta
        else:
            f0 = theta[:,0]
            alpha = theta[:,1]
            beta = theta[:,2]
            gamma = theta[:,3]
        ap1 = alpha+1;
        bp1 = beta+1;
        gp1 = gamma+1;

        teff = np.array([teff])
        teffFactor = self.getTeffFactor(teff)
        teffNorm = self.teffAverage(cs.tempRange, gamma)
        r = f0*((intXLimits[1]**ap1-intXLimits[0]**ap1)/(cs.periodRange[1]**ap1-cs.periodRange[0]**ap1)) \
            *((intYLimits[1]**bp1-intYLimits[0]**bp1)/(cs.rpRange[1]**bp1-cs.rpRange[0]**bp1)) \
            * teffFactor*(teff**gamma)/teffNorm
        return r

    def initRateModel(self):
        f0 = 0.8
        alpha = -0.8
        beta = -0.16
        gamma = 0
        theta = [f0, alpha, beta, gamma]
        return theta

# no fitted dependence on I, r or T
class constantFixedTeffAvg(populationModel):
    def __init__(self, cs = None):
        self.name = "constantFixedTeffAvg"
        self.bounds = [(0, 50000)]
        self.labels = [r"$F_0$"]
        self.teffBreak = 5117
        self.teffExp = [3.16, 4.49]
        self.teffNorm = [10**(-11.839), 10**(-16.769)]
        self.teffRef = 5778 # use the Sun for the normalization temperature
        
        self.norm = self.teffAverage(cs.tempRange)

    def teffAverage(self, intZLimits):
        ta0p1 = self.teffExp[0] + 1
        ta1p1 = self.teffExp[1] + 1

        if intZLimits[1] < self.teffBreak:
            teffInt = self.teffNorm[0]*(intZLimits[1]**ta0p1 - intZLimits[0]**ta0p1)/ta0p1
        elif intZLimits[0] > self.teffBreak:
            teffInt = self.teffNorm[1]*(intZLimits[1]**ta1p1 - intZLimits[0]**ta1p1)/ta1p1
        else:
            teffInt = self.teffNorm[0]*(self.teffBreak**ta0p1 - intZLimits[0]**ta0p1)/ta0p1 \
                        +self.teffNorm[1]*(intZLimits[1]**ta1p1 - self.teffBreak**ta1p1)/ta1p1
        return  teffInt/(intZLimits[1] - intZLimits[0])

    def getTeffFactor(self, teff):
        teffFactor = np.zeros(np.array(teff).shape)
        teffFactor[teff <= self.teffBreak] = self.teffNorm[0]*(teff[teff <= self.teffBreak])**self.teffExp[0]
        teffFactor[teff > self.teffBreak] = self.teffNorm[1]*(teff[teff > self.teffBreak])**self.teffExp[1]
        return teffFactor

    def rateModel(self, x, y, teff, xRange, yRange, tRange, theta):
        f0 = theta[0]
        
        teffFactor = self.getTeffFactor(teff)

        r = (f0/((xRange[1]-xRange[0])*(yRange[1]-yRange[0]))) * teffFactor/self.norm * np.ones(x.shape)
        return r
            
    def integrate(self, intXLimits, intYLimits, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0 = theta[0]
        else:
            f0 = theta[:,0]
        teffAvg = self.teffAverage(intZLimits)

        r = (f0*((intXLimits[1]-intXLimits[0])/(cs.periodRange[1]-cs.periodRange[0])) \
            *((intYLimits[1]-intYLimits[0])/(cs.rpRange[1]-cs.rpRange[0]))) \
            *teffAvg/self.norm
        return r
            
    def integrateZ(self, x, y, intZLimits, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0 = theta[0]
        else:
            f0 = theta[:,0]
        teffAvg = self.teffAverage(intZLimits)

        r = (f0/((cs.periodRange[1]-cs.periodRange[0])*(cs.rpRange[1]-cs.rpRange[0]))) * teffAvg/self.norm  * np.ones(x.shape)
        return r
                
    def integrateXY(self, intXLimits, intYLimits, teff, theta, cs):
#        f0, alpha, beta, gamma = theta
        if theta.ndim == 1:
            f0 = theta[0]
        else:
            f0 = theta[:,0]
        teff = np.atleast_2d(teff)
        
        teffFactor = self.getTeffFactor(teff)

        r = (f0*((intXLimits[1]-intXLimits[0])/(cs.periodRange[1]-cs.periodRange[0])) \
            *((intYLimits[1]-intYLimits[0])/(cs.rpRange[1]-cs.rpRange[0]))) \
            * teffFactor/self.norm
        return r

    def initRateModel(self):
        f0 = 0.8
        theta = [f0]
        return theta

class compSpace:
    def __init__(self, periodName, periodUnits, periodRange, nPeriod,
                 radiusName, radiusUnits, rpRange, nRp,
                tempName, tempUnits, tempRange, nTemp):
        self.periodName = periodName;
        self.periodUnits = periodUnits;
        self.periodRange = periodRange;
        self.nPeriod = nPeriod;
        self.radiusName = radiusName;
        self.radiusUnits = radiusUnits;
        self.rpRange = rpRange;
        self.nRp = nRp;
        self.tempName = tempName;
        self.tempUnits = tempUnits;
        self.tempRange = tempRange;
        self.nTemp = nTemp;
        
        self.period1D = np.linspace(self.periodRange[0], self.periodRange[1], self.nPeriod);
        self.rp1D = np.linspace(self.rpRange[0], self.rpRange[1], self.nRp);
        self.temp1D = np.linspace(self.tempRange[0], self.tempRange[1], self.nTemp);
#         self.temp1D = meanTeff
        self.period2D, self.rp2D = np.meshgrid(self.period1D, self.rp1D, indexing="ij");
        self.period2DTemp, self.temp2D = np.meshgrid(self.period1D, self.temp1D, indexing="ij");
        self.radius2DTemp, self.temp2D = np.meshgrid(self.rp1D, self.temp1D, indexing="ij");
        self.period3D, self.rp3D, self.temp3D = np.meshgrid(self.period1D, self.rp1D, self.temp1D, indexing="ij");
        self.vol = np.diff(self.period3D, axis=0)[:, :-1, :-1] \
                    * np.diff(self.rp3D, axis=1)[:-1, :, :-1] \
                    * np.diff(self.temp3D, axis=2)[:-1, :-1, :]
        self.vol2D = np.diff(self.period2D, axis=0)[:, :-1] \
                    * np.diff(self.rp2D, axis=1)[:-1, :]

def getHzFlux(teff, hzType = "optimistic"):
    if np.isscalar(teff):
        teff = np.array([teff])
    Ts = teff - 5780
#     # erratum Kopperapu 2013
#     KoppHzOptIn = 1.7763 + 1.4335e-4*Ts + 3.3954e-9*Ts**2 + -7.6364e-12*Ts**3 + -1.1950e-15*Ts**4
#     KoppHzInRunGreen = 1.0385 + 1.2456e-4*Ts + 1.4612e-8*Ts**2 + -7.6345e-12*Ts**3 + -1.711e-15*Ts**4
#     KoppHzPessIn = 1.0146 + 8.1884e-5*Ts + 1.9394e-9*Ts**2 + -4.3618e-12*Ts**3 + -6.8260e-16*Ts**4
#     KoppHzOut = 0.3507 + 5.9578e-5*Ts + 1.6707e-9*Ts**2 + -3.0058e-12*Ts**3 + -5.1925e-16*Ts**4

    if hzType == "optimistic":
        hzIndices = [0, 3]
    elif hzType == "conservative":
        hzIndices = [1, 2]
    else:
        raise ValueError('Bad catalog name');
    
    
    hzLabels = ["Recent Venus",
                "Runaway Greenhouse",
                "Maximum Greenhouse",
                "Early Mars",
                "Runaway Greenhouse for 5 ME",
                "Runaway Greenhouse for 0.1 ME"]
    seffsun  = [1.776,1.107, 0.356, 0.320, 1.188, 0.99]
    a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
    b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
    c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
    d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]

    hz = np.zeros((len(hzIndices), len(Ts)))
    for i in range(len(hzIndices)):
        hz[i,:] = seffsun[hzIndices[i]] + a[hzIndices[i]]*Ts + b[hzIndices[i]]*Ts**2 + c[hzIndices[i]]*Ts**3 + d[hzIndices[i]]*Ts**4
    
#     return KoppHzOptIn, KoppHzOut, KoppHzPessIn, KoppHzInRunGreen
    return hz

def hzOccRate(teffRange, rpRange, samples, model, cs, nSamples=None, hzType = "optimistic"):
    teffGrid = np.linspace(teffRange[0], teffRange[1], 1000)
    if samples.ndim == 1:
        hzOptOccRate = 0
    else:
        if nSamples == None:
            hzOptOccRate = np.zeros(samples.shape[0])
            useSamples = samples
        else:
            hzOptOccRate = np.zeros(nSamples)
            sampleIndex = np.floor(samples.shape[0]*np.random.rand(nSamples)).astype(int)
            useSamples = samples[sampleIndex,:]
    f = FloatProgress(min=0, max=len(teffGrid)-1)
    display(f)
    for t in range(len(teffGrid)-1):
        hz = getHzFlux(teffGrid[t], hzType)
        if samples.ndim == 1:
            hzOptOccRate += model.integrate([hz[1], hz[0]], rpRange, [teffGrid[t], teffGrid[t+1]], samples, cs)
            f.value += 1
        else:
            hzOptOccRate += model.integrate([hz[1], hz[0]], rpRange, [teffGrid[t], teffGrid[t+1]], useSamples, cs)
            f.value += 1
        
    return hzOptOccRate

def hzOccRate2D(teffRange, rpRange, samples, model, cs, nSamples=None, hzType = "optimistic", drawProgress = True):
    teff = np.linspace(teffRange[0], teffRange[1], np.round(teffRange[1]-teffRange[0]))
    hz = getHzFlux(teff, hzType)
    if drawProgress:
        f = FloatProgress(min=0, max=hz.shape[1])
        display(f)
    if samples.ndim == 1:
        hzOptOccRate = np.zeros(hz.shape[1])
        for s in range(hz.shape[1]):
            hzOptOccRate[s] = model.integrateXY([hz[1,s], hz[0,s]], rpRange, teff[s], samples, cs)
            if drawProgress:
                f.value += 1
    else:
        if nSamples != None:
            hzOptOccRate = np.zeros((hz.shape[1], nSamples))
        else:
            hzOptOccRate = np.zeros((hz.shape[1], samples.shape[0]))
        for s in range(hz.shape[1]):
            if nSamples != None:
                sampleIndex = np.floor(samples.shape[0]*np.random.rand(nSamples)).astype(int)
                useSamples = samples[sampleIndex,:]
            else:
                useSamples = samples
            hzOptOccRate[s,:] = model.integrateXY([hz[1,s], hz[0,s]], rpRange, teff[s], useSamples, cs)
            if drawProgress:
                f.value += 1

    return hzOptOccRate, teff


