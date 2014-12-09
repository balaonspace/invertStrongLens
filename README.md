invertStrongLens
================

MCMC method to invert strong gravitational lenses

REQUIREMENTS
python 3 + numpy + scipy


To simulate gravitationally lensed images for given lens and source parameters

python -W ignore GETimages.py psf.csv

The output is arc image (arcImage.csv) and simulated image (arcImage + noise, obsImage.csv)


To invert the given arc image for given lens parameters

python -W ignore GETchisqImage.py arcImage.csv obsImage.csv psf.csv

The output is source parameters, fitted image and chi-squared


To run the MCMC chain for given observed image and a random starting point for the lens parameters

python -W ignore RUNmcmc.py arcImage.csv obsImage.csv psf.csv

The output is posterior distribution for the lens and source parameters

