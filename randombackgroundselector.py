#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
import os
import shutil
import numpy as np
from os import path
from random import randint

# Quick script moves half of background files into backgrounds-to-merge folder

def move_half_to_merge():
    images_location = './backgrounds'
    output_location = './backgrounds-to-merge'
    if not os.path.isdir(output_location):
        raise FileNotFoundError(output_location)

    files = [f for f in os.listdir(images_location) if os.path.isfile(images_location + '/' + f)]
    no_of_files = len(files)
    # Done this way to prevent repeats in random_sample
    # Choosing random_sample(files, N) resulted in duplicates
    for x in range(no_of_files//2):
        random_sample = np.random.choice(files, 1)
        shutil.move(images_location + '/' + random_sample[0], output_location)
        files.remove(random_sample[0])
        print(random_sample[0], 'moved to', output_location)
    print(no_of_files//2, 'images moved to', output_location)
    print('Use Merge Images program to produce merged output.')

if __name__ == '__main__':
    move_half_to_merge()