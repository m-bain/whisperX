import os

root = "/project/shrikann_35/nmehlman/vpc"

for dirpath, dirnames, filenames in os.walk(root):
    if 'wav' in dirnames:
        print(dirpath)