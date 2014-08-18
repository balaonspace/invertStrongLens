#!/usr/bin/env python

#### INPUT - obsserved image, arc image, mask image, psf, number of samples for the MCMC analysis
#### OUTPUT - chi-squared between the observed image and fitted image, SSE for the fitted source profile

#### Randomly generate a large set of lens parameters and compute the chi-squared and SSE

import sys
import csv
import pyfits
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
import time
import matplotlib.pyplot as plt

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
	npix = kparam[1]	# Number of pixels - npix
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

	#### The deflection angle for a nonsingular isothermal ellipsoid
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
	#### this error can be ignored

	#### Return the magnification
	return mu



#### To invert the given arc and find the source flux, magnification
def getMisfitLensNIEXS(lensParam,arcImage,kparam,X,Xm):
	
	pixsz = kparam[0]
	
	#### The threshold for resampling the lens grid
	thresh = 30.0
	
	#### To find the magnification map for the given lens parameters

	#### First calculate the deflection angle and the rotated lens plane grid	
	Xr, alpha = getAlphaNIEXS(lensParam,Xm)
	#### Then calculate the Source plane grid
	Y = Xm - alpha
	#### The magnification map - mu
	mu = getMagnification(Y,pixsz)

	#### In the above magnification map the points near the critical curve are resampled and re-calculated
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
	mu = mu[arcImage>0]

	#### Compute the deflection angle for grid points in the lens plane

	Xr, alpha = getAlphaNIEXS(lensParam,X)

	#### Source plane grid
	Y = X - alpha

	#### Desired output parameters
	sourceData = Y[arcImage>0]
	imgData = arcImage[arcImage>0]
	sourceFlux = imgData/mu
	
	return sourceData, sourceFlux, magnification, Y 


#### The sersic function for the source parameters
def sersicFunc(r,n):
	return np.exp((-2.0*n+0.327)*(np.power(r,1/n)))


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


#### chiSq, SSE, indexn, Reffn, centroid, sc = getChisqImage(lensParam, kparam, obsImage, maskImage, sourceData, sourceFlux, magnification, Y)
#### Fit parameters for the source
#### find the images for the fitted source and lens parameters
#### obtain the chi-squared for the fitted image

def getChisqImage(lensParam, kparam, obsImage, maskImage, sourceData, sourceFlux, magnification, Y, psf):
	
	#### Find the max intensity and its position in the source plane
	maxInd = np.argmax(sourceFlux)
	maxIntensity = sourceFlux[maxInd] 
	
	#### choose the point of maximum intensity as peak and normalise the flux so that it is unit intensity at this point
	sourceFluxN = sourceFlux/maxIntensity

	#### Distance between the maximum intensity point and other points
	centroidSrc = sourceData[maxInd]
	r = np.abs(sourceData-sourceData[maxInd])
	
	#### Fit a sersic profile for the flux in the source plane
	step = np.max(r)/10.0
	ReffInitial = np.arange(0.5*np.max(r),np.max(r),step)

	sersicInd = np.zeros((5,1))
	SSE = np.zeros((5,1))

	for ii in range(5):
		ppot, pcov = sp.optimize.curve_fit(sersicFunc, r/ReffInitial[ii], sourceFluxN, p0=4.0)
		sersicInd[ii] = ppot
		sigmaSersicInd = pcov
		fitResidue = sersicFunc(r/ReffInitial[ii],sersicInd[ii]) - sourceFluxN
		#### SSE[ii] = np.sum(fitResidue*fitResidue)/fitResidue.shape[0]
		SSE[ii] = np.sum(fitResidue*fitResidue)

	sersicInd = sersicInd[np.argmin(SSE)]
	Reff = ReffInitial[np.argmin(SSE)]
	SSE = SSE[np.argmin(SSE)]
	
	kappaN = 2*sersicInd - 0.327
	Sc = maxIntensity/np.exp(kappaN)
	
	#### Fit the arcs for the given lens and source parameters
	#### getArcs(srcPos,Reff,Sc,sersicInd,kappaN,srcRad,srcPA,srcEllp,Y,magnification,lensPA)
	fittedImage = getArcs(centroidSrc,Reff,Sc,sersicInd,kappaN,np.max(r),0,0,Y,magnification,lensParam[3])
	
	#### convolve the fitted image with the PSF
	fittedImage = ndimage.convolve(fittedImage,psf,mode='constant',cval=0.0)

	#### chi squared for the fitted image
	#### division is by the standard deviation in the dominant noise in the image
	chiSq = np.sum((obsImage[maskImage==0]-fittedImage[maskImage==0])**2 / 8**2)


	return chiSq,SSE,sersicInd,Reff,centroidSrc,Sc,r,fittedImage




#########################################################################################################################
#### Load the arc image and observed image
#### The arc image is used for inversion
#### The observed image is used for chi squared subtraction
#### Load mask image if Lens is to be masked during chi squared evaluation

if len(sys.argv) != 6:
	sys.exit('Provide an arc image, observed image, mask image and psf image. If no mask image is present input a zero image. All images should have the same size')

arcImage = csv2array(sys.argv[1])
obsImage = csv2array(sys.argv[2])
maskImage = csv2array(sys.argv[3])
psf = csv2array(sys.argv[4])
nSamples = int(sys.argv[5])

#### The known information
#### known parameters 
#### kparam = [pixelSize imageSize fov1 fov2]
pixsz = 0.049
npix = arcImage.shape[0]

kparam = [pixsz, npix, 0, 0]
#### print kparam


#### Obtain the grid on the lens plane - X
x1, x2 = getLensGrid(kparam)
X = x1 + 1j*x2

#### Obtain the grid on the lens plane to compute magnification - Xm
kkparam = [pixsz, npix+1, 0, 0]
x1m, x2m = getLensGrid(kkparam)
Xm = x1m + 1j*x2m


#### The prior probability for the desired lens parameters
#### The prior probability is always chosen as a uniform distribution
#### Uniform distribution in numpy np.random.uniform(startValue, endValue, numberSamples)
#### LensParam = [einsteinRadius axisRatio coreRadius positionAngle xshear1 xshear2]
#### external shear is xshear = xshear1 + 1j*xshear2
#### short notation LensParam = [einRad f bc t xshear1 xshear2]
#### np.random.uniform(initialVal, finalVal, nSamples)
#### [1.5, 0.7, 0.3, pi/3, -0.03, 0.02]

einRad = np.random.uniform(1.2,1.8,nSamples)
#### The lower limit for axisRatio is 0.3 and upper limit is 0.99, there is singularity at f=1.
f = np.random.uniform(0.5,0.99,nSamples)
bc = np.random.uniform(0,0.5,nSamples)
t = np.random.uniform(pi/6,pi/2,nSamples)
xshear1 = np.random.uniform(-0.05,0.05,nSamples)
xshear2 = np.random.uniform(-0.05,0.05,nSamples)

lensParam = np.zeros((nSamples,6))
lensChi = np.zeros((nSamples,7))
for ii in range(nSamples):
	lensParam[ii,:] = [einRad[ii], f[ii], bc[ii], t[ii], xshear1[ii], xshear2[ii]]
	

#### Invert the images for the given set of lens parameters

tstart = time.time()

for jj in range(nSamples):
	
	#### Find the source image flux and magnification map
	sourceData, sourceFlux, magnification, Y = getMisfitLensNIEXS(lensParam[jj,:],arcImage,kparam,X,Xm)

	#### Find the chi-squared of the fitted image and the source parameters
	chiSq, SSE, indexn, Reffn, centroid, sc, r, fittedImage = getChisqImage(lensParam[jj,:], kparam, obsImage, maskImage, sourceData, sourceFlux, magnification, Y, psf)
	
	lensChi[jj,:] = [chiSq, SSE, indexn, Reffn, centroid.real, centroid.imag, sc]



#### Save the necessary output
#### The magnification map for the lens parameters
np.savetxt("lensPrior.csv",lensParam,delimiter=";")
np.savetxt("lensChi.csv",lensChi,delimiter=";")

print lensChi

tend = time.time() - tstart
print "Total time takes in seconds"
print tend
