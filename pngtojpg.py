#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
import os
from glob import glob
from PIL import Image
import numpy as np
import cv2

images_path = './merged/'

if not os.path.isdir(images_path):
    raise FileNotFoundError('Directory \'{}\' does not exist.'.format(images_path))

pngs = glob(images_path + '*.png')

for f in pngs:
    img = cv2.imread(images_path + f)
    cv2.imwrite(f[:-3] + 'jpg', img)
