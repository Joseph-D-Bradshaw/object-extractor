from os import listdir, mkdir
from os.path import isfile, isdir, join
import numpy as np
import json
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

# __author__ = "Joseph Bradshaw"
# __email__ = "joseph.bradshaw@outlook.com"
# __version__ = "1.0"


def checkSetup(inputPath, outputPath):
	"""Checks if program has necessary files/folders for program execution."""
	if not isdir(inputPath):
		raise FileNotFoundError(inputPath, 'No directory for images found')

	if not isdir(outputPath):
		mkdir(outputPath)

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


def main():
	inputPath = './images'
	outputPath = './'
	checkSetup(inputPath, outputPath)
	imageFiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f))]

	for imageName in imageFiles:
		imRGB = Image.open('{}/{}'.format(inputPath, imageName))
		imCV = cv2.imread('{}/{}'.format(inputPath, imageName))

		bgColour = getBgColour(imRGB)
		hsvColour = np.asarray(rgbToHsv(bgColour))
		hMin = hsvColour[0] - 5
		hMax = hsvColour[0] + 5
		sMin, vMin = 50, 50
		sMax, vMax = 255, 255

		BG_MIN = np.array([hMin, sMin, vMin], np.uint8)
		BG_MAX = np.array([hMax, sMax, vMax], np.uint8)

		imHSV = cv2.cvtColor(imCV, cv2.COLOR_BGR2HSV)
		
		print('bgColour {}:\n\tRGB {} | HSV {}'.format(imageName, bgColour, hsvColour))
		
		#TODO: Masking using cv2.inRange() to extract the background only
		mask = cv2.inRange(imHSV, BG_MIN, BG_MAX)
		# Background was grabbed - NOT it to grab what we want
		mask = cv2.bitwise_not(mask)
		cv2.imshow("Captured", mask)
		cv2.waitKey()
		cv2.destroyAllWindows()

		#cv2.imwrite('new-image5.jpg', outputImage)

main()
