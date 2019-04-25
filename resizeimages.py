#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
import os
from os.path import isfile, isdir, join
from PIL import Image
import numpy as np
import cv2
import time
import math
import multiprocessing

def resize_image(files, path_to_image):
    for f in files:
        img = cv2.imread(path_to_image + '/' + f)
        if img.shape[0] != 256 or img.shape[1] != 256:
            new_dim = (256, 256)
            img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_to_image + '/' + f, img)

images_path = './backgrounds'
if not os.path.isdir(images_path):
    raise FileNotFoundError('Directory \'{}\' does not exist.'.format(images_path))

NUM_OF_PROCESSES = 8

files = os.listdir(images_path)
files_changed = 0
num_of_files = len(files)
num_per_batch = math.ceil(len(files) / NUM_OF_PROCESSES)

batches = []
while len(files) > 0:
    batches.append(files[:num_per_batch])
    files = files[num_per_batch:]
del files

assert(len(batches) <= NUM_OF_PROCESSES), "Batches of jobs ({}) is not less than or equal to the max number of processes ({}).\nPer batch({})".format(len(batches), NUM_OF_PROCESSES, num_per_batch)

print('Files to process:', num_of_files)

jobs = []
start = time.time()
for i in range(len(batches)):
    p = multiprocessing.Process(target=resize_image, args=(batches[i], images_path))
    jobs.append(p)
    p.start()

for job in jobs:
    job.join()

end = time.time()
print('Time taken:', end-start, 'seconds')