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
#### import pyfits
#### import matplotlib.pyplot as plt

#### DEFINIG FUNCTIONS
pi = np.pi


#### To convert the given csv files to a numpy array or matrix
def csv2array(givenImage):
	arrayImage = []
	#givenCSV = csv.reader(open(givenImage,"rb"),delimiter=',')
	givenCSV = csv.reader(open(givenImage,"rt",encoding="utf8"),delimiter=',')
	
	for row in givenCSV:
		givenRow = []
		for item in row:
			givenRow.append(float(item))
		arrayImage.append(givenRow)
	return np.array(arrayImage)


#### To create a lens plane grid using the Telescope and CCD characteristics
def getLensGrid(kparam):
	pixsz = kparam[0]	# Pixel size in arc sec
	npix = kparam[1]	# Number of pixels
	x10 = kparam[2]
	x20 = kparam[3]
	
	if npix % 2:		# if npix is odd
		fov = np.arange(-(npix-1)/2,(npix+1)/2,1)*pixsz
	else:			# if npix is even
		fov = np.arange(-npix+1,npix+1,2)*pixsz/2

	x1range = x10+fov
	x2range = x20+fov
	x1, x2 = np.meshgrid(x1range,x2range)
	return x1,x2


#### To find the deflection angle in the Lens plane for given lens parameters
def getAlphaNIEXS(lensParam,X):
	einRad = lensParam[0]
	f = lensParam[1]
	fp = np.sqrt(1 - f*f)
	bc = lensParam[2]
	t = lensParam[3]
	xshear = (lensParam[4] + 1j*lensParam[5])
	
	#### The rotated coordinates	
	Xr = X * np.exp(-1j*t)
	x1 = Xr.real
	x2 = Xr.imag

	#### The impact parameter b
	b = x1 + 1j*f*x2
	#### The b-squared term
	bsq = np.absolute(b)
	bsq = np.multiply(bsq,bsq)
	#### The differentiation of bsq with respect to x
	bsqx = x1 + 1j*f*f*x2

	#### The deflection angle
	alpha = einRad*(np.sqrt(f)/fp)*( np.arctanh( fp * np.sqrt(bsq+(bc*bc)) / bsqx) - np.arctanh(fp*bc/(f*Xr)) )
	alpha = np.conj(alpha)*np.exp(1j*t) - xshear*np.conj(X)
	
	#### This part can return messages like
	#### "divide by zero encountered in divide"
	#### "invalid value encountered in divide"
	#### these messages can be ignored
	return Xr, alpha


#### The sersic function for the source parameters
def sersicFunc(r,Reff,n):
	return np.exp((-2.0*n+0.327)*(np.power(r/Reff,1/n)))

##### chiSq, SSE, indexn, Reffn, centroid, sc = getChisqImage(lensParam, kparam, obsImage, maskImage, sourceData, sourceFlux, magnification, Y)


#### To find the arc images for a given lens and source
def getArcs(sourceParam, Y):	
	
	Sc = sourceParam[0]
	sersicInd = sourceParam[1]
	kappaN = 2*sersicInd - 0.327
	Reff = sourceParam[2]
	srcRad = sourceParam[3]
	srcPos = sourceParam[4] + 1j*sourceParam[5]
	srcEllp = sourceParam[6]
	srcPA = sourceParam[7]
	
	npix = Y.shape[0]
	
	#### The center is adjusted so that it is traced to the image plane
	Yn = np.reshape(Y,(npix*npix,1))
	Yn1 = np.abs(Yn-srcPos)
	mi = np.where(Yn1-np.nanmin(Yn1) < 1e-6)
	Yn2 = Yn[mi]
	errc = Yn2 - srcPos

	#### The source plane is translated and rotated
	Ytr = np.exp(-1j*srcPA)*(Y - srcPos - errc)

	#### Hit parameter to constraint the points on the lens plane that have orginated from the source
	hit = (Ytr.real**2) + (Ytr.imag**2)/(1-srcEllp)**2
	#### np.savetxt("hit2.csv",hit,delimiter=",")
	hit[hit>srcRad**2] = 1e5

	#### Eliminate the core image
	hitCore = range(int(np.round(npix/2))-5,int(np.round(npix/2))+6)
	for ii in range(11):
		hit[hitCore[ii],hitCore] = 1e5
	
	#### np.savetxt("hit2.csv",hit,delimiter=",")

	#### Flux in the source plane
	fluxSrc = Sc*np.exp(-kappaN * (np.power(np.sqrt(hit)/Reff,1/sersicInd) - 1) )
	fluxSrc[fluxSrc<1e-3] = 0;
	
	return fluxSrc




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
sourceParam = [6, 4, 0.2, 0.4, -0.14, 0.14, 0.3, 0]

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
arcImage = sp.signal.convolve2d(arcImage,psf,mode='same')

obsImage = arcImage + noise 


arc = arcImage + noise 
arc[arcImage<16] = 0.0


#### Save the necessary output
np.savetxt("arcImage1.csv",arc,delimiter=",")
np.savetxt("obsImage1.csv",obsImage,delimiter=",")
