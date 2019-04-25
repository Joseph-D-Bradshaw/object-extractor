#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
import os
import shutil
import numpy as np
from os import path
from random import randint

# Quick script moves half of background files into another folder

images_location = '/home/soulskrix/Projects/Python/ObjectExtractor/backgrounds'
output_location = '/home/soulskrix/Projects/Python/ObjectExtractor/backgrounds-to-merge'
if not os.path.isdir(output_location):
    raise FileNotFoundError(output_location)

os.chdir(images_location)
files = [f for f in os.listdir(images_location) if os.path.isfile(f)]
no_of_files = len(files)
# Done this way to prevent repeats in random_sample
# Choosing random_sample(files, N) resulted in duplicates
for x in range(no_of_files//2):
    random_sample = np.random.choice(files, 1)
    shutil.move(images_location + '/' + random_sample[0], output_location)
    print(random_sample[0])
