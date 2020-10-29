import numpy as np
from scipy.integrate import romb
import matplotlib.pyplot as plt
from scipy.stats import gamma
import occRateModels as rm
from ipywidgets import FloatProgress
from IPython.display import display

def medianAndErrorbars(data):
    if data.ndim > 1:
        dataResult = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(data, [16, 50, 84],
                                                axis=0)))
        dataResult = list(dataResult)
        return dataResult
    else:
        v = np.percentile(data, [16, 50, 84])
        return [v[1], v[2]-v[1], v[1]-v[0]]


def printMedianAndErrorbars(data, precision=3):
    e = medianAndErrorbars(data)
    if data.ndim > 1:
        print("printMedianAndErrorbars only works for 1D arrays")
    else:
        formatStr = "{:." + str(precision) + "f}"
        return formatStr.format(e[0]) +"^{+" + formatStr.format(e[1]) + "}_{-" + formatStr.format(e[2]) + "}"


def integrate2DGrid(g, dx, dy):
    if g.shape[0]%2 == 0 or g.shape[1]%2 == 0:
        raise ValueError('integrate2DGrid requires a grid with odd number of points on a side');
    return romb(romb(g, dx), dy)

def integrateRateModel(periodRange, rpRange, theta, model, cs):
    nPts = 2**5+1 # must be 2**n + 1
    pGrid, rGrid = np.meshgrid(np.linspace(periodRange[0], periodRange[1], nPts),
                                       np.linspace(rpRange[0], rpRange[1], nPts),
                                       indexing="ij")
    dp = (pGrid[1,0]-pGrid[0,0])
    dr = (rGrid[0,1]-rGrid[0,0])
    
    if theta.ndim == 1:
        y = model.rateModel(pGrid, rGrid, cs.periodRange, cs.rpRange, theta)
        return integrate2DGrid(y, dp, dr)
    else: # assume first dimension is array of thetas
        ret = np.zeros(theta.shape[0])
        if len(ret) > 200:
            f = FloatProgress(min=0, max=len(ret))
            display(f)

        for i in range(len(ret)):
            y = model.rateModel(pGrid, rGrid, cs.periodRange, cs.rpRange, theta[i,:])
            ret[i] = integrate2DGrid(y, dp, dr)
            if len(ret) > 200:
                f.value += 1
        return ret


def integratePopTimesComp(periodRange, rpRange, theta, model, compGrid):
    nP = compGrid.shape[0]
    nR = compGrid.shape[1]

    pGrid, rGrid = np.meshgrid(np.linspace(periodRange[0], periodRange[1], nP),
                                       np.linspace(rpRange[0], rpRange[1], nR),
                                       indexing="ij")
    dp = (pGrid[1,0]-pGrid[0,0])
    dr = (rGrid[0,1]-rGrid[0,0])
    y = model.rateModel(pGrid, rGrid, period_rng, rp_rng, theta)*compGrid
    return integrate2DGrid(y, dp, dr)

class compSpace:
    def __init__(self, periodName, periodUnits, periodRange, nPeriod, radiusName, radiusUnits, rpRange, nRp):
        self.periodName = periodName;
        self.periodUnits = periodUnits;
        self.periodRange = periodRange;
        self.nPeriod = nPeriod;
        self.radiusName = radiusName;
        self.radiusUnits = radiusUnits;
        self.rpRange = rpRange;
        self.nRp = nRp;
        
        self.period1D = np.linspace(self.periodRange[0], self.periodRange[1], self.nPeriod);
        self.rp1D = np.linspace(self.rpRange[0], self.rpRange[1], self.nRp);
        self.period2D, self.rp2D = np.meshgrid(self.period1D, self.rp1D, indexing="ij");
        self.vol = np.diff(self.period2D, axis=0)[:, :-1] * np.diff(self.rp2D, axis=1)[:-1, :]

# population inference functions

def lnlike(theta, cs, koi_periods, koi_rps, summedCompleteness, model):
    pop = model.rateModel(cs.period2D, cs.rp2D, cs.periodRange, cs.rpRange, theta) * summedCompleteness
    pop = 0.5 * (pop[:-1, :-1] + pop[1:, 1:])
    norm = np.sum(pop * cs.vol)
    ll = np.sum(np.log(model.rateModel(koi_periods, koi_rps, cs.periodRange, cs.rpRange, theta))) - norm
    return ll if np.isfinite(ll) else -np.inf

# The ln-probability function is just propotional to the ln-likelihood
# since we're assuming uniform priors.
def lnprob(theta, cs, koi_periods, koi_rps, summedCompleteness, model):
    lp = lnPoisprior(theta, model)
    if not np.isfinite(lp):
        return -np.inf

    return lnlike(theta, cs, koi_periods, koi_rps, summedCompleteness, model)

# The negative ln-likelihood is useful for optimization.
# Optimizers want to *minimize* your function.
def nll(theta, cs, koi_periods, koi_rps, summedCompleteness, model):
    ll = lnlike(theta, cs, koi_periods, koi_rps, summedCompleteness, model)
    return -ll if np.isfinite(ll) else 1e15

def lnPoisprior(theta, model):
    bounds = model.getBounds()
    inRange = True;
    for i in range(len(bounds)):
        if (bounds[i][0] > theta[i]) | (theta[i] >= bounds[i][1]):
            inRange = False
    
    if inRange:
        return 1.0
    return -np.inf

# population analysis functions

# We'll reuse these functions to plot all of our results.
def make_plot(pop_comp, x0, x, y, ax):
#    print("in make_plot, pop_comp:")
#    print(pop_comp.shape)

    pop = 0.5 * (pop_comp[:, 1:] + pop_comp[:, :-1])
#    print("pop:")
#    print(pop.shape)
    pop = np.sum(pop * np.diff(y)[None, :, None], axis=1)
    a, b, c, d, e = np.percentile(pop * np.diff(x)[0], [2.5, 16, 50, 84, 97.5], axis=0)
    
    ax.fill_between(x0, a, e, color="k", alpha=0.1, edgecolor="none")
    ax.fill_between(x0, b, d, color="k", alpha=0.3, edgecolor="none")
    ax.plot(x0, c, "k", lw=1)

def plot_results(samples, cs, koiRps, koiPeriods, summedCompleteness, model, reversePeriod=False):
    # Loop through the samples and compute the list of population models.
    samples = np.atleast_2d(samples)
    pop = np.empty((len(samples), cs.period2D.shape[0], cs.period2D.shape[1]))
    gamma_earth = np.empty((len(samples)))

    for i, p in enumerate(samples):
        pop[i] = model.rateModel(cs.period2D, cs.rp2D, cs.periodRange, cs.rpRange, p)
        gamma_earth[i] = model.rateModel(365.25, 1.0, cs.periodRange, cs.rpRange, p) * 365.
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Integrate over period.
    dx = 0.25
    x = np.arange(cs.rpRange[0], cs.rpRange[1] + dx, dx)
    n, _ = np.histogram(koiRps, x)
    
    fsize = 18
    # Plot the observed radius distribution.
    ax = axes[0, 0]
    make_plot(pop * summedCompleteness[None, :, :], cs.rp1D, x, cs.period1D, ax)
    ax.errorbar(0.5*(x[:-1]+x[1:]), n, yerr=np.sqrt(n), fmt=".k",
                capsize=0)
    ax.set_xlim(cs.rpRange[0], cs.rpRange[1])
    ax.set_xlabel(cs.radiusName + " [" + cs.radiusUnits + "]", fontsize = fsize)
    ax.set_ylabel("# of detected planets", fontsize = fsize)
    
    # Plot the true radius distribution.
    ax = axes[0, 1]
    make_plot(pop, cs.rp1D, x, cs.period1D, ax)
    ax.set_xlim(cs.rpRange[0], cs.rpRange[1])
    ax.set_ylim(0, 0.37)
    ax.set_xlabel(cs.radiusName + " [" + cs.radiusUnits + "]", fontsize = fsize)
    ax.set_ylabel("$\mathrm{d}N / \mathrm{d}R$; $\Delta R = 0.25\,R_\oplus$", fontsize = fsize)
    
    # Integrate over period.
#    dx = 31.25
    dx = (cs.periodRange[1] - cs.periodRange[0])/11
    x = np.arange(cs.periodRange[0], cs.periodRange[1] + dx, dx)
    n, _ = np.histogram(koiPeriods, x)
    
    # Plot the observed period distribution.
    ax = axes[1, 0]
    make_plot(np.swapaxes(pop * summedCompleteness[None, :, :], 1, 2), cs.period1D, x, cs.rp1D, ax)
    ax.errorbar(0.5*(x[:-1]+x[1:]), n, yerr=np.sqrt(n), fmt=".k",
                capsize=0)
    if reversePeriod:
        ax.set_xlim(cs.periodRange[1], cs.periodRange[0])
    else:
        ax.set_xlim(cs.periodRange[0], cs.periodRange[1])
#    ax.set_ylim(0, 79)
    yl = ax.get_ylim()
    ax.set_ylim(0, 1.3*yl[1])
    ax.set_xlabel(cs.periodName + " [" + cs.periodUnits + "]", fontsize = fsize)
    ax.set_ylabel("# of detected planets", fontsize = fsize)
    
    # Plot the true period distribution.
    ax = axes[1, 1]
    make_plot(np.swapaxes(pop, 1, 2), cs.period1D, x, cs.rp1D, ax)
    if reversePeriod:
        ax.set_xlim(cs.periodRange[1], cs.periodRange[0])
    else:
        ax.set_xlim(cs.periodRange[0], cs.periodRange[1])
    ax.set_ylim(0, 0.27)
    ax.set_xlabel(cs.periodName + " [" + cs.periodUnits + "]", fontsize = fsize)
    ax.set_ylabel("$\mathrm{d}N / \mathrm{d}P$; $\Delta P = 31.25\,\mathrm{days}$", fontsize = fsize)
    
                
    return gamma_earth, fig
