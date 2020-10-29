# DR25-occurrence-public
This repository contains the code used to compute the results reported in a series of three papers on computing the occurrence of rocky exoplanets in their star's habitable zone, using the Kepler DR25 catalog.  These papers are

"A Probabilistic Approach to Kepler Completeness and Reliability for Exoplanet Occurrence Rates" https://iopscience.iop.org/article/10.3847/1538-3881/ab8a30, https://arxiv.org/abs/1906.03575, which develops new techniques for correcting for vetting completeness and reliability.  This paper computes illustrative occurrence rates that show the impact of reliability correction, but those occurrence rates are not expected to be accurate (as described in the paper).  This paper uses the code in the directories GKbaseline*.  

"Reliability Correction is Key for Robust Kepler Occurrence Rates" https://iopscience.iop.org/article/10.3847/1538-3881/abb316, https://arxiv.org/abs/2006.15719, which demonstrates the occurrence rates are robustly independent of catalog variations when correcting for reliability, and depend on catalog variations when not correcting for reliability.  This paper uses the code in the directory GKRobovetterVariations.

"The Occurrence of Rocky Habitable Zone Planets Around Solar-Like Stars from Kepler Data" (to appear in The Astronomical Journal), https://arxiv.org/abs/2010.14812, which computes occurrence rates using the flux incidenct on exoplanets, carefully selecting parameters optimal for habitability.  Unlike the previous two papers, we have confidence in the accuracy of the occurrence rates presented in this paper.  This paper uses the code in the directory insolation.

This repository also contains the code used for the paper "Exoplanet Occurrence Rates of Mid M-dwarfs Based on Kepler DR25", https://iopscience.iop.org/article/10.3847/2515-5172/ab7afd, which does a simple estimate of the occurrence of exoplanets around mid M-dwarfs.  This paper uses the code in the directory midMDwarfs.

For a more gentle introduction to the techniques used in this code, see the repository DR25-occurrence-tutorial.

The code in this repository is not intended to be used by the community, and is neither maintained nor supported.  It is research code that is not intended to be user-friendly.  It is presented so the community can examine how we got the results in the above papers. 
