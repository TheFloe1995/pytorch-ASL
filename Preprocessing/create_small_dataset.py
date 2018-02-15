import os, glob
from os.path import relpath
from shutil import copyfile

# This script creates a small dataset out of the big Fingerspelling one (all 5 speakers, all letters but less pics per speaker)
if __name__ == '__main__':

    ############## Type in your dataset5 path here #####
    dir_orig = '/Datasets/dataset5'
    dir_small = '/Datasets/dataset_half'

    ####################################################

    # all (nested) subdirectorys
    dirs = [relpath(x[0], dir_orig) for x in os.walk(dir_orig)]

    if not os.path.exists(dir_small):
        os.mkdir(dir_small)

        for dir in dirs:
            if not dir == '.':
                os.mkdir(dir_small+'/'+dir)

            for filename in glob.glob(dir_orig+'/'+dir+"/color_*_???[0,2,4,6,8].png"):
                filename_rel = relpath(filename, dir_orig)
                try:
                    copyfile(filename, dir_small+'/'+filename_rel)
                except:
                    print('No file named '+ filename_rel)

    else:
        print('dataset_small folder already exists. Please delete it before creating it again.')
