#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
from os import listdir, mkdir, remove
from os.path import isfile, isdir, join
from PIL import Image
import math
import numpy as np
import cv2
import multiprocessing
import time


# __author__ = "Joseph Bradshaw"
# __email__ = "joseph.bradshaw@outlook.com"


def check_setup(input_path, output_path, bg_path=None):
	"""Checks if program has necessary files/folders for program execution."""
	if not isdir(input_path):
		raise FileNotFoundError(input_path, 'No directory for images found')

	if not isdir(output_path):
		mkdir(output_path)

	if bg_path:
		if not isdir(bg_path):
			raise FileNotFoundError(bg_path, 'No directory for background images found')	

def get_bg_color(PIL_image):
	"""Retrieves background colour by most common colour pixel detection."""
	pixel_count, detected_bg_color = max(PIL_image.getcolors(PIL_image.size[0] * PIL_image.size[1]))
	return detected_bg_color

def rgb_to_hsv(rgb):
	"""Converts rgb to OpenCV hsv ranges."""
	r,g,b = rgb
	r,g,b = r/255, g/255, b/255
	
	c_max = max(r, g, b)
	c_min = min(r, g, b)
	c_delta = c_max - c_min

	if c_delta == 0:
		hue = 0
	elif c_max == r:
		hue = 60 * (((g-b)/c_delta) % 6)
	elif c_max == g:
		hue = 60 * ((b-r)/c_delta+2)
	else:
		hue = 60 * ((r-g)/c_delta+4)
	hue = int(round(hue))

	if c_max == 0:
		sat = 0
	else:
		sat = c_delta/c_max

	val = c_max

	# HSV originally H=0-360, S=0-100, V=0-100
	# Convert to OpenCV HSV H=0-180, S=0-255, V=0-255
	hue //= 2
	sat = int(sat * 255)
	val = int(val * 255)

	return (hue, sat, val)

def get_hsv_thresholds(hsv_color):
	"""Returns min and max thresholds as (hsv_min, hsv_max) when provided (hue, sat, val)"""
	hue_min = hsv_color[0] - 5
	hue_max = hsv_color[0] + 5
	sat_min, val_min = 50, 50
	sat_max, val_max = 255, 255
	hsv_min = np.array([hue_min, sat_min, val_min], np.uint8)
	hsv_max = np.array([hue_max, sat_max, val_max], np.uint8)
	return (hsv_min, hsv_max)

def denoise_mask(mask):
	kernel = np.ones((5,5), np.uint8)
	dilation = cv2.dilate(mask, kernel, iterations=1)
	erosion = cv2.erode(dilation, kernel, iterations=1)
	return erosion

def process_images(input_path, output_path, images_names, suppress_output=True, LIVE_VIEW=False):
	for image_name in images_names:
		# Detect bg_color and sort into hsv min/max thresholds
		image_RGB = Image.open('{}/{}'.format(input_path, image_name))
		image_CV = cv2.imread('{}/{}'.format(input_path, image_name))
		bg_color = get_bg_color(image_RGB)
		hsv_color = np.asarray(rgb_to_hsv(bg_color))
		hsv_thresholds = get_hsv_thresholds(hsv_color)

		if not suppress_output:
			print('bg_color {}:\n\tRGB {} | HSV {}'.format(image_name, bg_color, hsv_color))

		# Select background and create inverse to select object as masks
		image_HSV = cv2.cvtColor(image_CV, cv2.COLOR_BGR2HSV)
		mask_bg = cv2.inRange(image_HSV, hsv_thresholds[0], hsv_thresholds[1])
		mask_fg = cv2.bitwise_not(mask_bg)

		denoised_mask = denoise_mask(mask_fg)
		contours, hier = cv2.findContours(denoised_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		largest_contour = max(contours, key=cv2.contourArea)
		(x,y,w,h) = cv2.boundingRect(largest_contour)
		capture_mask = np.zeros(mask_fg.shape, dtype=np.uint8)
		capture_mask[y:y+h, x:x+w] = mask_fg[y:y+h, x:x+w]

		# Create BGRA empty transparent image for merging with object image
		image_empty = np.zeros((image_CV.shape[0], image_CV.shape[1], 4), dtype=np.uint8)
		image_CV = cv2.cvtColor(image_CV, cv2.COLOR_BGR2BGRA)
		output_image = cv2.bitwise_or(image_empty, image_CV, mask=capture_mask)

		# Crop final output and ave image to disk
		output_image = output_image[y:y+h, x:x+w]
		out = '{}/{}{}'.format(output_path, image_name[:-4], '-processed.png')
		cv2.imwrite(out, output_image)

		# Display for demo
		hspace = 0
		padding = 20

		original_window = 'Original Image'
		cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(original_window, (image_CV.shape[1]//3, image_CV.shape[0]//3))
		cv2.moveWindow(original_window, 0, 0)
		cv2.imshow(original_window, image_CV)
		hspace += image_CV.shape[1]//3 + padding

		contour_window = 'Largest Contour Capture'
		contour_copy = image_CV.copy()
		cv2.drawContours(contour_copy, [largest_contour], 0, (0,0,255), 3)
		cv2.namedWindow(contour_window, cv2.WINDOW_NORMAL)        
		cv2.resizeWindow(contour_window, (denoised_mask.shape[1]//3, denoised_mask.shape[0]//3))
		cv2.moveWindow(contour_window, hspace, 0)
		cv2.imshow(contour_window, contour_copy)
		hspace += denoised_mask.shape[1]//3 + padding

		background_window = 'Background Mask'
		cv2.namedWindow(background_window, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(background_window, (mask_bg.shape[1]//3, mask_bg.shape[0]//3))
		cv2.moveWindow(background_window, hspace, 0)
		cv2.imshow(background_window, mask_bg)
		hspace += mask_bg.shape[1]//3 + padding

		foreground_window = 'Foreground Mask'
		cv2.namedWindow(foreground_window, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(foreground_window, (mask_fg.shape[1]//3, mask_fg.shape[0]//3))
		cv2.moveWindow(foreground_window, hspace, 0)
		cv2.imshow(foreground_window, mask_fg)
		hspace += mask_fg.shape[1]//3 + padding

		output_window = 'Processed Output'
		cv2.namedWindow(output_window, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(output_window, (output_image.shape[1]//3, output_image.shape[0]//3))
		cv2.moveWindow(output_window, hspace, 0)
		cv2.imshow(output_window, output_image)
		hspace += 256 + padding

		cv2.waitKey()
		cv2.destroyAllWindows()

def start_extracting():
	print('Object Extractor start.. ', end='')

	input_path = './images'
	output_path = './processed'
	check_setup(input_path, output_path)

	NUM_OF_PROCESSES = 1

	image_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
	num_of_images = len(image_files)
	num_per_batch = math.ceil(len(image_files) / NUM_OF_PROCESSES)
	batches = []
	while len(image_files) > 0:
		batches.append(image_files[:num_per_batch])
		image_files = image_files[num_per_batch:]
	del image_files
	
	assert(len(batches) <= NUM_OF_PROCESSES), "Batches of jobs ({}) is not less than or equal to the max number of processes ({}).\nPer batch({})".format(len(batches), NUM_OF_PROCESSES, num_per_batch)

	jobs = []
	start = time.time()
	for i in range(len(batches)):
		p = multiprocessing.Process(target=process_images, args=(input_path, output_path, batches[i], False, True))
		jobs.append(p)
		p.start()

	for job in jobs:
		job.join()

	end = time.time()
	print("{} images processed from \'{}\', results can be found in \'{}\'.".format(num_of_images, input_path, output_path))
	print("Complete in", end-start, "seconds.")
	print("Delete any defective images and place into './cleaned-processed' before merging.")

if __name__ == '__main__':
	start_extracting()