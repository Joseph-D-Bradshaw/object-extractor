#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
from os import listdir, mkdir, remove
from os.path import isfile, isdir, join
from PIL import Image
import numpy as np
import cv2
#import multiprocessing
import time
from random import randint
import math
import multiprocessing
from objectextractor import check_setup

def merge_images(fg_batch, bg_batch, fg_path, bg_path, output_path, output_name, merge_num):
    assert(len(fg_batch) == len(bg_batch)), "Foreground and background batches are not of equal length fg: {} vs bg: {}".format(len(fg_batch), len(bg_batch))

    for i in range(len(fg_batch)):
        bg_image = Image.open(bg_path + '/' + bg_batch[i])
        fg_image = Image.open(fg_path + '/' + fg_batch[i])
    
        bg_width, bg_height = bg_image.size
        shrink_val = randint(2,8)
        max_size = (bg_width//shrink_val, bg_height//shrink_val)
        
        thumb = fg_image.copy()
        thumb.thumbnail(max_size, Image.ANTIALIAS)
        fg_width, fg_height = thumb.size

        x = randint(0, bg_width)
        y = randint(0, bg_height)
        if x + fg_width > bg_width:
            x -= fg_width
        if y + fg_height > bg_height:
            y -= fg_height

        bg_image.paste(thumb, (x,y), thumb)
        bg_image.save(output_path + '/' + output_name + '{}.jpg'.format(merge_num))
        merge_num += 1


input_path = './cleaned-processed'
output_path = './merged'
bg_path = './backgrounds-to-merge'
check_setup(input_path, output_path, bg_path)
image_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
bg_files = [f for f in listdir(bg_path) if isfile(join(bg_path, f))]
assert(len(image_files) == len(bg_files)), "Number of files to merge (bg and fg) must match. fg: {} vs bg: {}".format(len(image_files), len(bg_files))

NUM_OF_PROCESSES = 8

num_of_images = len(image_files)
num_per_batch = math.ceil(len(image_files) / NUM_OF_PROCESSES)
fg_batches = []
while len(image_files) > 0:
    fg_batches.append(image_files[:num_per_batch])
    image_files = image_files[num_per_batch:]
del image_files

bg_batches = []
while len(bg_files) > 0:
    bg_batches.append(bg_files[:num_per_batch])
    bg_files = bg_files[num_per_batch:]
del bg_files

jobs = []
image_number = 1
start = time.time()
for i in range(len(fg_batches)):
    p = multiprocessing.Process(target=merge_images, args=(fg_batches[i], bg_batches[i], input_path, bg_path, output_path, 'robot', image_number))
    jobs.append(p)
    p.start()
    image_number += len(fg_batches[i])

for job in jobs:
    job.join()

end = time.time()

print('Time taken:', end-start)

# for i, image_name in enumerate(image_files):
#     # Background info
#     bg_image = Image.open(bg_path + '/' + bg_files[i])
#     bg_width, bg_height = bg_image.size
#     # Thumbnail info
#     shrink_val = randint(2, 8)
#     max_size = (bg_width//shrink_val, bg_height//shrink_val)
#     image_fg = Image.open(input_path + '/' + image_files[i])
#     thumb = image_fg.copy()
#     thumb.thumbnail(max_size, Image.ANTIALIAS)
#     obj_width, obj_height = thumb.size
#     # Location info
#     x = randint(0, bg_width)
#     y = randint(0, bg_height)
#     if x + obj_width > bg_width:
#         x -= obj_width
#     if y + obj_height > bg_height:
#         y -= obj_height

#     bg_image.paste(thumb, (x, y), thumb)
#     bg_image.save(output_path + '/' + 'robot{}.jpg'.format(i+1))
#     print(i+1, (x,y))

# end = time.time()