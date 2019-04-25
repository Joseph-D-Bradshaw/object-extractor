import shutil
import os

src_path = './merged/'
src = os.listdir(src_path)
dst = input('Enter path to destination folder: ')
if not os.path.isdir(dst):
    raise FileNotFoundError("Input destination does not exist.")

n = int(input('Enter number of files you want to move: '))
if not isinstance(n, int):
    raise ValueError("Only integer values accepted for number of items to move.")

c = 0
for f in src:
    if c == n:
        break
    print(src_path + f)
    if os.path.isfile(src_path+f):
        shutil.move(src_path+f, dst)
    c += 1