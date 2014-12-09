#!/usr/bin/env python

#### For a given arc image and observed image and lens parameters
#### Find the source parameters and solve for the images corresponding to the source and lens parameters
#### Obtain the chi-squared between the observed image and the fitted image

import sys
import csv
### import pyfits
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
from scipy import signal
import time
### import matplotlib.pyplot as plt

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


#### To invert the given arc and find the source flux, magnification
def getMisfitLensNIEXS(lensParam,arcImage,X):
	
	Xr, alpha = getAlphaNIEXS(lensParam,X)

	#### Source plane grid
	Y = X - alpha

	#### Desired output parameters
	sourceData = Y[arcImage>0]
	sourceFlux = arcImage[arcImage>0]
		
	return sourceData, sourceFlux, Y 


#### The sersic function for the source parameters
def sersicFunc(r,n):
	return np.exp((-2.0*n+0.327)*(np.power(r,1/n)))


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

##### chiSq, SSE, indexn, Reffn, centroid, sc = getChisqImage(lensParam, kparam, obsImage, maskImage, sourceData, sourceFlux, magnification, Y)
#### Fit parameters for the source
#### find the images for the fitted source and lens parameters
#### obtain the chi-squared for the fitted image

def getChisqImage(obsImage, maskImage, sourceData, sourceFlux, Y, sN, fRe,psf):
	
	#### Find the max intensity and its position in the source plane
	maxInd = np.argmax(sourceFlux)
	maxIntensity = sourceFlux[maxInd] 
	
	#### choose the point of maximum intensity as peak and normalise the flux so that it is unit intensity at this point
	sourceFluxN = sourceFlux/maxIntensity

	#### Distance between the maximum intensity point and other points
	centroidSrc = sourceData[maxInd]
	r = np.abs(sourceData-sourceData[maxInd])
	R = np.max(r)
	
	#### Fix the effective radii and the radius of the source
	Reffn = fRe*R
	rSrc = 2*Reffn
	
	#### Fit a sersic profile for the flux in the source plane
	ppot, pcov = sp.optimize.curve_fit(sersicFunc, r/Reffn, sourceFluxN, p0=4.0)
	sersicInd = ppot
	sigmaSersicInd = pcov
	fitResidue = sersicFunc(r/Reffn,sersicInd) - sourceFluxN
	SSE = np.sum(fitResidue*fitResidue)
		
		
	kappaN = 2*sersicInd - 0.327
	Sc = maxIntensity/np.exp(kappaN)
	
	#### Fit the arcs for the given lens and source parameters
	#### sourceParam = [Intensity SersicIndex reff rSrc y1Src y2Src ellpSrc paSrc]
	sourceParam = [Sc, sersicInd, Reffn, rSrc, centroidSrc.real, centroidSrc.imag, 0, 0]
	#### sourceParam = [Sc, 1.3*sersicInd, Reffn, rSrc, centroidSrc.real, centroidSrc.imag, 0, 0]
	fittedImage = getArcs(sourceParam, Y)
	fittedImage = sp.signal.convolve2d(fittedImage,psf,mode='same')
	
	#### chi squared for the fitted image
	#### division is by the standard deviation in the dominant noise in the image
	#chiSq = np.sum((obsImage[maskImage>0 or fittedImage > sN]-fittedImage[maskImage>0 | fittedImage > sN])**2 / sN**2)
	chiSq = np.sum(( obsImage[np.logical_or(maskImage>0, fittedImage>sN)]-fittedImage[np.logical_or(maskImage>0, fittedImage>sN)])**2 / sN**2)
	
	#chiSq, SSE, Sc, indexn, Reffn, rSrc, centroid, R, r, fittedImage
	return chiSq,SSE,Sc,sersicInd,Reffn,rSrc,centroidSrc.real,centroidSrc.imag,R,r,fittedImage

#########################################################################################################################
#### Load the arc image and observed image
#### The arc image is used for inversion
#### The observed image is used for chi squared subtraction
#### Load mask image if Lens is to be masked during chi squared evaluation

if len(sys.argv) != 4:
	sys.exit('Provide an arc image, observed image and psf image. The arc image and observed image should be of same size.')

arcImage = csv2array(sys.argv[1])
obsImage = csv2array(sys.argv[2])
maskImage = csv2array(sys.argv[1])
psf = csv2array(sys.argv[3])

#### Factor used in the denominator of the chi-squared calculation
NM = np.sum(arcImage>0)

#### The known information
#### known parameters 
#### kparam = [pixelSize imageSize fov1 fov2]
pixsz = 0.049
npix = arcImage.shape[0]

kparam = [pixsz, npix, 0, 0]

#### LensParam = [einsteinRadius axisRatio coreRadius positionAngle xshear1 xshear2]
#### external shear is xshear = xshear1 + 1j*xshear2
#### short notation LensParam = [einRad f bc t xshear1 xshear2]

#### Invert the lens for the given set of lens parameters

lensParam = [1.6, 0.8, 0.0, pi/4.5, 0.00, 0.00]

#### Other inputs
#### sN - the standard deviation in noise
sN = 8
#### fRe - the factor by which the distance between the center and farthest point of the reconstructed source is multiplied to obtain the effective radii
fRe = 1

#### Obtain the grid on the lens plane - X
x1, x2 = getLensGrid(kparam)
X = x1 + 1j*x2

#### Find the source image flux and magnification map
sourceData, sourceFlux, Y = getMisfitLensNIEXS(lensParam,arcImage,X)

#### Find the chi-squared of the fitted image and the source parameters
chiSq, SSE, Sc, indexn, Reffn, rSrc, centroidReal, centroidImag, R, r, fittedImage = getChisqImage(obsImage, maskImage, sourceData, sourceFlux, Y, sN, fRe, psf)


#### lensChi = [chiSq/NM, SSE, Sc, indexn, Reffn, rSrc, centroidReal, centroidImag, R]
print("Chisquared for the fitted image = ", chiSq/NM)
print("Summed squared error for source = ", SSE)
print("Source Parameters")
print("Intensity at effective radii = ", Sc)
print("Sersic index = ", indexn)
print("Effective radii = ", Reffn)
print("Radius of the source = ", rSrc)
print("Centroid = ", centroidReal, " + ", centroidImag,"j")


#### Save the necessary output
np.savetxt("fitImage1.csv",fittedImage,delimiter=",")
np.savetxt("radius.csv",r,delimiter=",")
np.savetxt("srcFlux.csv",sourceFlux,delimiter=",")


