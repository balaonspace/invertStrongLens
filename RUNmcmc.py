#!/usr/bin/env python


#### To find the chi-squared for the fitted image using a set of randomly generated lens parameters

import sys
import csv
### import pyfits
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
from scipy import signal
import time
import invertSLmcmc
from invertSLmcmc import *

#########################################################################################################################
#### Load the arc image and observed image
#### The arc image is used for inversion
#### The observed image is used for chi squared subtraction
#### Load mask image if Lens is to be masked during chi squared evaluation

if len(sys.argv) != 4:
	sys.exit('Provide an arc image, observed image, mask image and psf image. If no mask image is present input a zero image. All images should have the same size')

obsImage = csv2array(sys.argv[1])	
arcImage = csv2array(sys.argv[2])
maskImage = csv2array(sys.argv[2])
psf = csv2array(sys.argv[3])

NM = np.sum(arcImage>0)

#### The known information
#### known parameters 
#### kparam = [pixelSize imageSize fov1 fov2]
pixsz = 0.049
npix = arcImage.shape[0]

kparam = [pixsz, npix, 0, 0]

#### Obtain the grid on the lens plane - X
x1, x2 = getLensGrid(kparam)
X = x1 + 1j*x2

#### MCMC sample size
nsamples = 1000
#### Store every M'th sample
M = 100

#### Uniform distribution in numpy np.random.uniform(startValue, endValue, numberSamples)
#### LensParam = [einsteinRadius axisRatio coreRadius positionAngle xshear1 xshear2]
#### external shear is xshear = xshear1 + 1j*xshear2
#### short notation LensParam = [einRad f bc t xshear1 xshear2]
#### Initial lens parameters
lensParam = np.array([1.7, 0.75, 0, 0, 0, 0])

#### fixed size jump parameters for lens
jumpParamL = np.array([0.03, 0.05, 0, pi/10, 0, 0])

#### upper and lower bound on the lens parameters
Lub = np.array([1.9, 0.99, 0, pi/2, 0, 0])
Llb = np.array([1.5, 0.5, 0, 0, 0, 0])

#### uncertainty in flux data
#### sN - the standard deviation in noise
sN = 11

centroid, Imax, R = getCentroid(lensParam,arcImage,X)
#### optimise the initial parameters
p = lensParam
print(p)
for ii in range(10000):
	if (R > 0.5*p[0]):
		p = lensParam + np.multiply(2*np.random.rand(6)-1,jumpParamL)
		centroid, Imax, R = getCentroid(p,arcImage,X)
	else:
		lensParam = p
		break
		


#### Initial parameters for the source
sourceParam = np.array([0, 4, 0, R, centroid.real, centroid.imag, 0, 0])

#### fixed size jump parameters for the source
jumpParamS = np.array([0, 0.65, 0, 0, 0, 0, 0, 0])

#### upper and lower bound for the source parameters
Sub = np.array([0,7,0,0.5*lensParam[0], 0, 0.5, 0.5, 0, 0])
Slb = np.array([0,0.5,0,0, 0, -0.5, -0.5, 0, 0])

lensP, sourceP = getMCMClens(obsImage,arcImage,maskImage,X,lensParam,sourceParam,sN,psf,jumpParamL,jumpParamS,Lub,Llb,Sub,Slb,NM,nsamples,M,Imax)

maxInd = np.argmax(sourceP[:,6])

print (lensP)
print (sourceP)
print("Lens Parameters")
print(lensP[maxInd,:])
print("Source Parameters")
print(sourceP[maxInd,:])

print("Lens Parameters")
print (np.mean(lensP,axis=0))
print (np.std(lensP,axis=0))
print("Source Parameters")
print (np.mean(sourceP,axis=0))
print (np.std(sourceP,axis=0))
