#!/usr/bin/env python

#### To obtain the images for a given set of lens parameters and source parameters
#### The lens is a non singular isothermal ellipsoid with external shear
#### The source has a sersic profile

import sys
import csv
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
from scipy import signal
import time
import invertSLmcmc
from invertSLmcmc import *
#### import pyfits
#### import matplotlib.pyplot as plt

#### DEFINIG FUNCTIONS
pi = np.pi

########################################################################################################################
##### THE ROUTINE STARTS HERE

if len(sys.argv) != 2:
	sys.exit('Provide a psf file in csv format.')
psf = csv2array(sys.argv[1])

#### The known information
#### known parameters 
#### kparam = [pixelSize imageSize fov1 fov2]
pixsz = 0.049
npix = 151

kparam = [pixsz, npix, 0, 0]

#### LensParam = [einsteinRadius axisRatio coreRadius positionAngle xshear1 xshear2]
#### external shear is xshear = xshear1 + 1j*xshear2
#### short notation LensParam = [einRad f bc t xshear1 xshear2]
#### sourceParam = [Intensity sersicIndex effectiveRad srcRad srcPos1 srcPos2 srcEllp srcPA]
#### srcEllp = 0, circular source

lensParam = [1.6, 0.8, 0.0, pi/4.5, 0.00, 0.00]
sourceParam = [3.2229, 4, 0.243, 4*0.243, -0.14, 0.14, 0.3, 0]

print ("Given Lens Parameters")
print ("Einstein Radius ", lensParam[0])
print ("Axis Ratio ", lensParam[1])
print ("Core Radius ", lensParam[2])
print ("Position angle ", lensParam[3])
print ("External shear1 ", lensParam[4])
print ("External shear2 ", lensParam[5])
print ("")
print ("Given Source Parameters")
print ("Intensity ", sourceParam[0])
print ("Sersic index ", sourceParam[1])
print ("Effective radii ", sourceParam[2])
print ("Total radii ", sourceParam[3])
print ("Position 1 ", sourceParam[4])
print ("Position 2 ", sourceParam[5])

#### Add noise to the arc image
#### noise = np.random.normal(mean,standardDeviation,(xSize,ySize))
noise = np.random.normal(0,8,(npix,npix))

#### Obtain the grid on the lens plane - X
x1, x2 = getLensGrid(kparam)
X = x1 + 1j*x2

#### Find the source image flux and magnification map
Xr, alpha = getAlphaNIEXS(lensParam,X)
Y = X - alpha

arcImage = getArcs(sourceParam, Y)

#### PSF convolve the simulated image
arcImage = sp.signal.convolve2d(arcImage,psf,mode='same')

#### add noise to the PSF convolved image
obsImage = arcImage + noise 


arc = arcImage + noise 
arc[arcImage<22] = 0.0


#### Save the necessary output
np.savetxt("arcImage1.csv",arc,delimiter=",")
np.savetxt("obsImage1.csv",obsImage,delimiter=",")
