# SODA_Python
This repository consists in a python version of SODA algorithm [1]. With the addition of some distance metrics.

##  Distance Metrics ##
### Offline Mode
##### - Magnitude
 - Euclidean: straight line between two points;
 - Mahalanobis: Multi-dimensional generalization of how many standard deviations away a point is from another;
 - Cityblock: Distance between two vectors if they could only move right angles (taxicab/manhattan);
 - Chebyshev: The greatest of difference between two vectors along any coordinate dimension;
 - Minkowski: Generalization of other distances dependent of a parameter $p$ (in this code $p=1.5$ ):
      - p = 1 $\rightarrow$ cityblock;
      - p = 2 $\rightarrow$ euclidean;
      - p = $\infty$ $\rightarrow$ chebyshev.
 - Canberra: Weighted version of Cityblock, the distinction is that the absolute difference between the variables of the two objects is divided by the sum of the absolute variable values prior to summing. It's more sensitive for points close to origin.
 
##### - Angular
 - Cossine Dissimilarity: Is one minus the cosine of the angle between two vectors
  
### Online Mode
##### - Magnitude
 - Euclidean: straight line between two points;
 
##### - Angular
 - Cossine Dissimilarity: Is one minus the cosine of the angle between two vectors

# Folders Layout

### IPython
Python Notebook version of SODA
 - SODA.ipynb 
      - Offline version
 - SODA_Online.ipynb
      - Offline + Online version
 - exampledata.csv
      - Two columns example data

### MATLAB
Original Version of SODA
 - SelfOrganisedDirectionAwareDataPartitioning.m
      - Internal functions
 - demo_SODA_hybrid.m
      - Offline + Online SODA
 - demo_SODA_offline.m
      - Offline SODA
 - exampledata.mat
      - Two columns example data

### Python
Python Version of SODA
 - SODA.py
      - Script version of SODA Algorithm
 - SODA_numba.py
      - Script version of SODA Algorithm with Numba applied in order to reduce execution time

# References
[1] - X. Gu, P. Angelov, D. Kangin, J. Principe, Self-organised direction aware data partitioning algorithm, Information Sciences, vol.423, pp. 80-95 , 2018.