#### Functions for the MCMC routine

import sys
import csv
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import ndimage
from scipy import signal
import time

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

	
#### Find the area and position vector of the source	
def getCentroid(lensParam,arcImage,X):
	
	Xr, alpha = getAlphaNIEXS(lensParam,X)

	#### Source plane grid
	Y = X - alpha

	#### Desired output parameters
	sourceData = Y[arcImage>0]
	sourceFlux = arcImage[arcImage>0]
	
	#### Find the max intensity and its position in the source plane
	maxInd = np.argmax(sourceFlux)
	maxIntensity = sourceFlux[maxInd] 

	#### Distance between the maximum intensity point and other points
	centroid = sourceData[maxInd]
	r = np.abs(sourceData-sourceData[maxInd])
	R = np.max(r)
		
	return centroid, maxIntensity, R

#### get chisquared for the given lens and source
def getChisquared(X,obsImage,maskImage,lensParam,sourceParam,sN,psf):
	Xr, alpha = getAlphaNIEXS(lensParam,X)
	Y = X - alpha
	
	obsFit = getArcs(sourceParam,Y)
	obsFit = sp.signal.convolve2d(obsFit,psf,mode='same')
	
	#### chisquared
	chiSq = np.sum(( obsImage[np.logical_or(maskImage>0, obsFit>sN)]-obsFit[np.logical_or(maskImage>0, obsFit>sN)])**2 / sN**2)
	#print (chiSq)
	return chiSq,obsFit	

	
#### MCMC routine for the source
def getMCMCsource(obsImage, maskImage, lensParam, sourceParam, psf, X, sN, jumpParamS, Sub, Slb, NM, Imax, nsamples,M):
	#### intensity at effective radius
	Ie = Imax/np.exp(2*sourceParam[1]-0.327)
	sourceParam[0] = 3*Ie*np.random.rand() + Ie
	
	#### effective radii
	sourceParam[2] = np.random.rand()*sourceParam[3]
	chisq, uk = getChisquared(X,obsImage,maskImage,lensParam,sourceParam,sN,psf)
	
	#### likelihood for the source parameter
	likelihoodO = -abs(chisq/NM)
	
	Src = np.zeros((nsamples/M,7))
	Src[0,:] = [sourceParam[0], sourceParam[1], sourceParam[2], sourceParam[3], sourceParam[4], sourceParam[5], likelihoodO]
	
	for kk in range(nsamples):
		s = sourceParam + np.multiply(2*np.random.rand(8)-1,jumpParamS)
		Ie = Imax/np.exp(2*s[1]-0.327)
		s[0] = 3*Ie*np.random.rand() + Ie
		s[2] = np.random.rand()*s[3]
		
		if (s[1]>Slb[1] and s[1]<Sub[1]):
			chisq, uk = getChisquared(X,obsImage,maskImage,lensParam,s,sN,psf)
			
			likelihoodP = -abs(chisq/NM)
			
			alpha = np.min([1.0, np.exp(likelihoodP-likelihoodO)])
			
			if np.random.rand() < alpha:
				likelihoodO = likelihoodP
				sourceParam = s
				
		
		if kk%M == 0:
			Src[kk/M,:] = [sourceParam[0], sourceParam[1], sourceParam[2], sourceParam[3], sourceParam[4], sourceParam[5], likelihoodO]
		
	return Src


#### MCMC routine for the lens	
def getMCMClens(obsImage,arcImage,maskImage,X,lensParam,sourceParam,sN,psf,jumpParamL,jumpParamS,Lub,Llb,Sub,Slb,NM,nsamples,M,Imax):
	
	#### the bounds for the lens
	bb = [0,1,3]
	Llb1 = Llb[bb]
	Lub1 = Lub[bb]
	
	#### get the source parameters for the initial lens parameters
	Src = getMCMCsource(obsImage, maskImage, lensParam, sourceParam, psf, X, sN, jumpParamS, Sub, Slb, NM, Imax, 100,10)
	maxInd = np.argmax(Src[:,6])
	sourceParam = Src[maxInd,:]
	likelihoodO = sourceParam[6]
	
	#### initialise the MCMC
	lensP = np.zeros((nsamples/M,6))
	sourceP = np.zeros((nsamples/M,7))
	lensP[0,:] = [lensParam[0], lensParam[1], lensParam[2], lensParam[3], lensParam[4], lensParam[5]]
	sourceP[0,:] = [sourceParam[0], sourceParam[1], sourceParam[2], sourceParam[3], sourceParam[4], sourceParam[5], sourceParam[6]]

	for kk in range(nsamples):
		#### random walk for the lens parameter
		p = lensParam + np.multiply(2*np.random.rand(6)-1,jumpParamL)
	
		centroid, Imax, R = getCentroid(lensParam,arcImage,X)
		srcParam = [0, 4, 0, R, centroid.real, centroid.imag, 0, 0]
		
		pp = p[bb]
		if (sum(pp>Llb1) == 3 and sum(pp<Lub1) == 3 and R<0.5*p[0]):
			
			Src = getMCMCsource(obsImage, maskImage, p, srcParam, psf, X, sN, jumpParamS, Sub, Slb, NM, Imax, 100,10)
			maxInd = np.argmax(Src[:,6])
			srcParam = Src[maxInd,:]
			likelihoodP = srcParam[6]
			
			alpha = np.min([1.0, np.exp(likelihoodP-likelihoodO)])
			
			if np.random.rand() < alpha:
				lensParam = p
				likelihoodO = likelihoodP
				sourceParam = srcParam
	
		if kk%M == 0:
			lensP[int(kk/M),:] = [lensParam[0], lensParam[1], lensParam[2], lensParam[3], lensParam[4], lensParam[5]]
			sourceP[int(kk/M),:] = [sourceParam[0], sourceParam[1], sourceParam[2], sourceParam[3], sourceParam[4], sourceParam[5], sourceParam[6]]

			
	return lensP, sourceP


	
