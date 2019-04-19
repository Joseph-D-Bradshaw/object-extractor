#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
from os import listdir, mkdir, remove
from os.path import isfile, isdir, join
from PIL import Image
import numpy as np
import cv2
import multiprocessing
import time


# __author__ = "Joseph Bradshaw"
# __email__ = "joseph.bradshaw@outlook.com"


def checkSetup(inputPath, outputPath, bgPath=None):
	"""Checks if program has necessary files/folders for program execution."""
	if not isdir(inputPath):
		raise FileNotFoundError(inputPath, 'No directory for images found')

	if not isdir(outputPath):
		mkdir(outputPath)

	if bgPath:
		if not isdir(bgPath):
			raise FileNotFoundError(bgPath, 'No directory for background images found')	


def getBgColour(PILImage):
	"""Retrieves background colour by most common colour pixel detection."""
	pixelCount, detectedBgColor = max(PILImage.getcolors(PILImage.size[0] * PILImage.size[1]))
	return detectedBgColor

def rgbToHsv(rgb):
	"""Converts rgb to OpenCV hsv ranges."""
	r,g,b = rgb
	r,g,b = r/255, g/255, b/255
	
	cMax = max(r, g, b)
	cMin = min(r, g, b)
	cDelta = cMax - cMin

	if cDelta == 0:
		hue = 0
	elif cMax == r:
		hue = 60 * (((g-b)/cDelta) % 6)
	elif cMax == g:
		hue = 60 * ((b-r)/cDelta+2)
	else:
		hue = 60 * ((r-g)/cDelta+4)
	hue = int(round(hue))

	if cMax == 0:
		sat = 0
	else:
		sat = cDelta/cMax

	val = cMax

	# HSV originally H=0-360, S=0-100, V=0-100
	# Convert to OpenCV HSV H=0-180, S=0-255, V=0-255
	hue //= 2
	sat = int(sat * 255)
	val = int(val * 255)

	return (hue, sat, val)

def processImages(inputPath, outputPath, imageNames, suppressOutput=True):
	for imageName in imageNames:
		imRGB = Image.open('{}/{}'.format(inputPath, imageName))
		imCV = cv2.imread('{}/{}'.format(inputPath, imageName))
		# Calculate HSV thresholds for colour detection
		bgColour = getBgColour(imRGB)
		hsvColour = np.asarray(rgbToHsv(bgColour))
		hMin = hsvColour[0] - 5
		hMax = hsvColour[0] + 5
		sMin, vMin = 50, 50
		sMax, vMax = 255, 255
		# Define min and max HSV thresholds
		BG_MIN = np.array([hMin, sMin, vMin], np.uint8)
		BG_MAX = np.array([hMax, sMax, vMax], np.uint8)
		if not suppressOutput:
			print('bgColour {}:\n\tRGB {} | HSV {}'.format(imageName, bgColour, hsvColour))
		# Select background and create inverse to select object as masks
		imHSV = cv2.cvtColor(imCV, cv2.COLOR_BGR2HSV)
		maskBgd = cv2.inRange(imHSV, BG_MIN, BG_MAX)
		maskFgd = cv2.bitwise_not(maskBgd)
		# Clean up mask by only choosing to include largest contour and denoise
		kernel = np.ones((5,5), np.uint8)
		dilation = cv2.dilate(maskFgd, kernel, iterations=1)
		erosion = cv2.erode(dilation, kernel, iterations=1)
		contours, hier = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		largestContour = max(contours, key=cv2.contourArea)
		(x,y,w,h) = cv2.boundingRect(largestContour)
		objCaptureMask = np.zeros(maskFgd.shape, dtype=np.uint8)
		objCaptureMask[y:y+h, x:x+w] = maskFgd[y:y+h, x:x+w]
		# Create BGRA empty transparent image for merging with object image
		imEmpty = np.zeros((imCV.shape[0], imCV.shape[1], 4), dtype=np.uint8)
		imCV = cv2.cvtColor(imCV, cv2.COLOR_BGR2BGRA)
		outputImage = cv2.bitwise_or(imEmpty, imCV, mask=objCaptureMask)
		# Crop final output
		outputImage = outputImage[y:y+h, x:x+w]
		# Save image to disk
		out = '{}/{}{}'.format(outputPath, imageName[:-4], '-processed.png')
		cv2.imwrite(out, outputImage)

if __name__ == '__main__':
	print('Object Extractor start.. ', end='')
	inputPath = './images'
	outputPath = './processed'
	checkSetup(inputPath, outputPath)
	imageFiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f))]

	mid = len(imageFiles) // 2
	start = mid // 2
	end = mid + start

	batch1 = imageFiles[:start]
	batch2 = imageFiles[start:mid]
	batch3 = imageFiles[mid:end]
	batch4 = imageFiles[end:]

	p1 = multiprocessing.Process(target=processImages, args=(inputPath, outputPath, batch1))
	p2 = multiprocessing.Process(target=processImages, args=(inputPath, outputPath, batch2))
	p3 = multiprocessing.Process(target=processImages, args=(inputPath, outputPath, batch3))
	p4 = multiprocessing.Process(target=processImages, args=(inputPath, outputPath, batch4))

	print('Jobs started..')
	start = time.time()
	p1.start()
	p2.start()
	p3.start()
	p4.start()

	p1.join()
	p2.join()
	p3.join()
	p4.join()
	end = time.time()
	print("{} images processed from \'{}\', results can be found in \'{}\'.".format(len(imageFiles), inputPath, outputPath))
	print("Complete in", end-start, "seconds.")