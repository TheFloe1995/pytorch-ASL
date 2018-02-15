import os, glob

# This script deletes all depth images (filename is starting with depth*) in a given folder
# Modify the path to the path where all depth images should be removed (as well in all subfolders)

if __name__ == '__main__':

    ################# Type in your dataset5 path here #################################################
    ####### CHOOSE THIS PATH CAREFULLY AS ALL FILES STARTING WITH 'DEPTH' IN THIS PATH WILL BE DELETED#
    directory = '../../../dataset5'
    ###################################################################################################

    #all (nested) subdirectorys
    dirs = [x[0] for x in os.walk(directory)]
    print(dirs)

    for dir in dirs:
        print('dir is '+ dir)
        for filename in glob.glob(dir+"/depth*"):
            print('file to be removed: '+ filename)
            os.remove(filename)