import sys
import os, glob
from os.path import relpath
from shutil import copyfile

# This script takes a speaker directory as input and splits for all subdirectory folders images into two batches.
# The two batches will contain images from first half of original set and second half respectively.
# Two new pseudo speaker folders will be created to store the batches in. These folders will be structured as the original one.
# Write the command line argument WITHOUT a slash at the end.
if not len(sys.argv) == 2:
    raise ValueError('Please give exactly one command line argument (path of speaker folder, where letters are split).')

dir_orig = sys.argv[1]
dir1 = dir_orig+'1'
dir2 = dir_orig + '2'

if not os.path.exists(dir_orig):
    raise ValueError("This path does not exist. Please choose a valid one")

if not os.path.exists(dir1) and not os.path.exists(dir2):
    os.mkdir(dir1)
    os.mkdir(dir2)

    subdirs_orig = [relpath(x[0], dir_orig) for x in os.walk(dir_orig)]

    for dir in subdirs_orig:
        if not dir == '.':
            os.mkdir(dir1 + '/' + dir)
            os.mkdir(dir2 + '/' + dir)

            all_files = glob.glob(dir_orig+'/'+dir+"/*.png")
            all_files = sorted(all_files, key=str.lower)
            num_files = len(all_files)

            for i in range(num_files):
                filename = all_files[i]
                filename_rel = relpath(filename, dir_orig)

                if i+1 <= num_files/2:  #i+1: due to zero-indexing
                    copyfile(filename, dir1 + '/' + filename_rel)
                else:
                    copyfile(filename, dir2 + '/' + filename_rel)


else:
    print('Folder {0} or {1} already exists. If you nevertheless want to create a new split, please remove the directory.'.format(dir1,dir2))

