# DR25-occurrence
Python notebooks that explain the computation of various DR25 occurrence rate products

This directory contains the notebooks and python code used to compute the GK baseline.  The flow of computation is:

To make completeness contours:
1) run binomialVettingCompleteness.ipynb
2) go to ../completenessContours and run compute_num_completeness_mproc.py using vetCompletenessTable.pkl as the 11th input, with the name of the fit function in the 12th input

To compute the reliability data and prepare the PC population
0) select a model function (such as "rotatedLogisticX0") and set the model variables in the following before running
1) run binomialObsFPRate.ipynb to fit the observed FP rate
2) run binomialFPEffectiveness.ipynb to fit the FP effectiveness
3) run makePlanetInput.ipynb uses the observed FP rate and FP effectiveness to compute false alarm reliability, applies FPP to compute total reliability, and selects the planet population

To compute the occurrence rate
1) run computeOccurrence.ipynb
