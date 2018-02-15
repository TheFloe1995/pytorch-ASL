import os, glob
from os.path import relpath
from PIL import Image, ImageOps


# This script creates a small dataset out of the big Fingerspelling one (all 5 speakers, all letters but less pics per speaker)
if __name__ == '__main__':

    ############## Type in your dataset5 path here #####
    dir_orig = '../../../dataset5'
    ####################################################

    #this directory will be created
    dir_black_padded = dir_orig + '/../dataset_5_b'
    # all (nested) subdirectorys
    dirs = [relpath(x[0], dir_orig) for x in os.walk(dir_orig)]

    if not os.path.exists(dir_black_padded):
        os.mkdir(dir_black_padded)

        for dir in dirs:
            print(dir)
            if not dir == '.':
                os.mkdir(dir_black_padded + '/' + dir)

            for filename in glob.glob(dir_orig+'/'+dir+"/*.png"):
                filename_rel = relpath(filename, dir_orig)

                im = Image.open(filename)
                w, h = im.size

                max_len = max(w,h)
                pad = abs(w-h)//2
                im_b = ImageOps.expand(im,border = pad)
                ww,hh = im_b.size

                if h>w:
                    im_b = im_b.crop((0,pad,ww,hh-pad))
                else:
                    im_b = im_b.crop((pad,0,ww-pad,hh))

                im_b.save(dir_black_padded+'/'+filename_rel[:-4]+'_b.png')

    else:
        print('dataset_small folder already exists. Please delete it before creating it again.')