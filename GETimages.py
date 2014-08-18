#!/usr/bin/env python

#### To obtain the images for a given set of lens parameters and source parameters
#### The lens is a non singular isothermal ellipsoid
#### The source has a sersic light profile

import sys
import csv
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
import time
#### import pyfits
#### import matplotlib.pyplot as plt

#### DEFINIG FUNCTIONS
pi = np.pi


#### To convert the given csv files to a numpy array or matrix
def csv2array(givenImage):
	arrayImage = []
	givenCSV = csv.reader(open(givenImage,"rb"),delimiter=',')

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



#### To calculate the magnification
def getMagnification(Y,pixsz):
	#### the size of the image
	n = Y.shape[0]-1
	
	area = pixsz*pixsz*np.ones((n,n))
	areaQ = np.ones((n,n))
	
	#### The real and imaginary part of Y in the source plane
	Yreal = Y.real
	Yimag = Y.imag

	#### The square pixels on the lens plane are mapped to quadrilateral pixels in the source plane
	#### The area enclosed by a quadrilateral can be written as the sum of two triangles
	for ii in range(n):
		for jj in range(n):
			#### Area of first triangle
			T1 = [ [1.0,1.0,1.0], [Yreal[ii,jj],Yreal[ii,jj+1],Yreal[ii+1,jj]], [Yimag[ii,jj],Yimag[ii,jj+1],Yimag[ii+1,jj]] ]
			areaT1 = np.abs(np.linalg.det(T1))

			#### Area of second triangle
			T2 = [ [1.0,1.0,1.0], [Yreal[ii+1,jj+1],Yreal[ii,jj+1],Yreal[ii+1,jj]], [Yimag[ii+1,jj+1],Yimag[ii,jj+1],Yimag[ii+1,jj]] ]
			areaT2 = np.abs(np.linalg.det(T2))
			
			#### Area of the quadrilateral
			areaQ[ii,jj] = areaT1+areaT2
	
	#### magnification map - mu
	mu = area/areaQ
	#### Setting an upper threshold on the magnification
	mu[mu>100.0] = 100.0
	#### Above command might return message
	#### "invalid value encountered in greater"

	#### Return the magnification
	return mu



#### To invert the given arc and find the source flux, magnification
def getSourcePlane(lensParam,kparam,X,Xm):
	
	pixsz = kparam[0]
	
	#### The threshold for my above which you need to resample the grid
	thresh = 30.0
	
	#### To find the magnification map for the given lens parameters

	#### First calculate the deflection angle and the rotated lens plane grid	
	Xr, alpha = getAlphaNIEXS(lensParam,Xm)
	#### Then calculate the Source plane grid
	Y = Xm - alpha
	#### The magnification map - mu
	mu = getMagnification(Y,pixsz)

	#### In the above magnification map the points near the critical curve are re-sampled and re-calculated
	m = mu
	x1 = X.real
	x2 = X.imag
	
	x1 = x1[m > thresh]
	x2 = x2[m > thresh]
	m = m[m > thresh]
	muIndex = np.array(np.nonzero(mu > thresh))

	hpixsz = pixsz/2 	#### half pixel size
	

	for ii in range(m.shape[0]):
		kkparam = [hpixsz,3,x1[ii],x2[ii]]
		
		XX1,XX2 = getLensGrid(kkparam)
		XX = XX1 + 1j*XX2
		
		XXr, alphaCrit = getAlphaNIEXS(lensParam,XX)
		YY = XX - alphaCrit
		
		muCrit = np.sum(getMagnification(YY,hpixsz))/4.0
		mu[muIndex[0,ii],muIndex[1,ii]] = muCrit	

	magnification = mu

	#### Compute the deflection angle for grid points in the lens plane

	Xr, alpha = getAlphaNIEXS(lensParam,X)

	#### Source plane grid
	Y = X - alpha
	
	return magnification, Y 


#### The sersic function for the source parameters
def sersicFunc(r,Reff,n):
	return np.exp((-2.0*n+0.327)*(np.power(r/Reff,1/n)))


#### To find the arc images for a given lens and source
def getArcs(srcPos,Reff,Sc,sersicInd,kappaN,srcRad,srcPA,srcEllp,Y,magnification,lensPA):	

	npix = Y.shape[0]
	
	#### The sourc plane is tranlated and rotated
	Ytr = np.exp(-1j*srcPA)*(Y - srcPos)

	#### Hit parameter to constraint the points on the lens plane that have orginated from the source
	hit = (Ytr.real**2) + (Ytr.imag**2)/(1-srcEllp)**2
	#### np.savetxt("hit2.csv",hit,delimiter=",")
	hit[hit>srcRad**2] = 1e5

	#### Eliminate the core image
	hitCore = range(np.round(npix/2)-5,np.round(npix/2)+6)
	for ii in range(11):
		hit[hitCore[ii],hitCore] = 1e5
	
	#### np.savetxt("hit2.csv",hit,delimiter=",")

	#### Flux in the source plane
	fluxSrc = Sc*np.exp(-kappaN * (np.power(np.sqrt(hit)/Reff,1/sersicInd) - 1) )
	#fluxSrc = (np.power(np.sqrt(hit)/Reff,1/sersicInd) )

	#### np.savetxt("flux2.csv",fluxSrc,delimiter=",")
	#### Observed image or arcs 
	arcsObs = np.multiply(fluxSrc,magnification)
	#arcsObs = fluxSrc*magnification
	
	return arcsObs




########################################################################################################################
##### THE ROUTINE STARTS HERE

#### The known information
#### known parameters 
#### kparam = [pixelSize imageSize fov1 fov2]
pixsz = 0.049
npix = 151

kparam = [pixsz, npix, 0, 0]
#### print kparam


#### Obtain the grid on the lens plane - X
x1, x2 = getLensGrid(kparam)
X = x1 + 1j*x2

#### Obtain the grid on the lens plane to compute magnification - Xm
kkparam = [pixsz, npix+1, 0, 0]
x1m, x2m = getLensGrid(kkparam)
Xm = x1m + 1j*x2m


#### LensParam = [einsteinRadius axisRatio coreRadius positionAngle xshear1 xshear2]
#### external shear is xshear = xshear1 + 1j*xshear2
#### short notation LensParam = [einRad f bc t xshear1 xshear2]
#### sourceParam = [srcPos1 srcPos2 srcRad effectiveRad srcEllp srcPA]
#### srcEllp = 0, circular source

#### Invert the lens for the given set of lens parameters and source parameters

#### lensParam = [1.5, 0.7, 0.3, pi/3, -0.03, 0.02]

lensParam = [1.5, 0.8, 0.2, pi/3, 0.1, 0.00]
srcParam = [0.0, 0.0, 0.4, 0.3, 0, 0]

srcPos = srcParam[0]+1j*srcParam[1]
srcRad = srcParam[2]
Reff = srcParam[3]
srcPA = srcParam[4]
srcEllp = srcParam[5]
sersicInd = 4.0
Sc = 1.0
kappaN = 2.0*sersicInd - 0.327

lensPA = lensParam[3]

#### Find the source image flux and magnification map
magnification, Y = getSourcePlane(lensParam,kparam,X,Xm)

arcImage = getArcs(srcPos,Reff,Sc,sersicInd,kappaN,srcRad,srcPA,srcEllp,Y,magnification,lensPA)
### arcImage[arcImage<40] = 0.0

#### Add noise to the arc image
#### sigma + np.random.randn(dimension1,dimension2) + mean

#### noise = 8*np.random.randn(151,151)
noise = np.random.normal(4,8,(151,151))
obsImage = arcImage + noise 

axisLim = npix*pixsz/2
##im1 = plt.imshow(obsImage, interpolation='bilinear',origin='lower',extent=[-axisLim,axisLim,-axisLim,axisLim])
#plt.show()


arcImage[arcImage<40] = 0.0


#### Save the necessary output
#### The magnification map for the lens parameters
np.savetxt("magnification.csv",magnification,delimiter=",")
np.savetxt("arcImage.csv",arcImage,delimiter=",")
np.savetxt("obsImage.csv",obsImage,delimiter=",")
